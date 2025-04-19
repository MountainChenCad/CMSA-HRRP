# scripts/compute_centers.py
import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from logger import loggers
from utils import set_seed, normalize, load_config, get_dynamic_paths # Import helpers

logging.basicConfig(level=logging.INFO) # Basic config for early messages
logger = logging.getLogger(__name__)

def main(config_path: str):
    """Computes mean visual features for base classes using a pre-trained adapter and VLM."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup Logging ---
    log_dir = dynamic_paths['centers_log_dir']
    os.makedirs(log_dir, exist_ok=True)
    log = loggers(os.path.join(log_dir, 'compute_centers_log.txt'))
    log.info("Starting base class center computation...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"VLM Variant: {config['model']['foundation_model']['variant']}")
    log.info(f"Text Type (used for adapter training): {config['semantics']['generation']['text_type']}")
    log.info(f"Centers will be saved to: {dynamic_paths['base_centers_path']}")
    log.info(f"Logs will be saved to: {log_dir}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Base Dataset ---
    log.info("Loading base dataset...")
    try:
        sim_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'simulated_hrrp')
        meas_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'measured_hrrp')
        base_dataset = HRRPDataset(
            root_dirs=[sim_path, meas_path],
            split='base',
            classes=config['data']['base_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude' # Match adapter input expectation
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}", exc_info=True)
         sys.exit(1)

    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['batch_size'], # Reuse batch size from config
        shuffle=False, # No need to shuffle
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load VLM Visual Encoder ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path): log.error(f"VLM weights not found: {vlm_weights_path}"); sys.exit(1)
        log.info(f"Loading VLM weights from: {vlm_weights_path}")

        vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        visual_encoder = vlm_model.visual.to(device).eval() # f_V
        for param in visual_encoder.parameters():
            param.requires_grad = False
        log.info("VLM Visual Encoder loaded and frozen.")
        expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                           (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
        if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
    except Exception as e:
        log.error(f"Failed to load VLM: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Pre-trained Adapter ---
    # Use the adapter corresponding to the current config (VLM variant, text_type)
    adapter_checkpoint_path = dynamic_paths['adapter_latest_ckpt'] # Or use 'adapter_best_ckpt'
    log.info(f"Loading pre-trained adapter from: {adapter_checkpoint_path}")
    try:
        if not os.path.exists(adapter_checkpoint_path):
            log.error(f"Adapter checkpoint file not found at {adapter_checkpoint_path}. Run adapter training first for this config.")
            sys.exit(1)

        adapter_config = config['model'].get('adapter_1d_to_2d', {})
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'],
            input_channels=1, output_channels=3, output_size=expected_img_size,
            intermediate_dim=adapter_config.get('intermediate_dim', 2048)
        ).to(device)
        adapter_checkpoint_data = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint_data['adapter_state_dict'])
        adapter.eval()
        log.info(f"Pre-trained adapter loaded and frozen (from epoch {adapter_checkpoint_data.get('epoch', 'N/A')}).")
    except Exception as e:
        log.error(f"Error loading adapter checkpoint: {e}", exc_info=True)
        sys.exit(1)

    # --- Feature Extraction ---
    log.info("Extracting visual features (z_V) for all base samples...")
    all_features = []
    all_labels = []
    visual_dim = config['model']['foundation_model']['visual_encoder_dim'] # Get expected dim
    with torch.no_grad():
        for hrrp_samples, labels in tqdm(base_loader, desc="Extracting Features"):
            hrrp_samples = hrrp_samples.to(device)
            try:
                pseudo_images = adapter(hrrp_samples)
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)

                visual_features = visual_encoder(pseudo_images) # z_V
                if isinstance(visual_features, tuple): visual_features = visual_features[0]
                if visual_features.dim() == 3 and visual_features.shape[1] > 1:
                     visual_features = visual_features[:, 0] # CLS token

                # Verify dimension
                if visual_features.shape[-1] != visual_dim:
                    log.warning(f"Extracted feature dim {visual_features.shape[-1]} != config dim {visual_dim}. Check VLM/config.")
                    # Decide how to handle: error, skip, project? For centers, best to error out or use config dim.
                    log.error("Dimension mismatch, cannot reliably compute centers. Exiting.")
                    sys.exit(1)

                # Store UNNORMALIZED features for computing the mean center (as per SemFew)
                all_features.append(visual_features.cpu())
                all_labels.append(labels.cpu())
            except Exception as e:
                 log.error(f"Error during feature extraction: {e}. Skipping batch.", exc_info=True)
                 continue

    if not all_features:
         log.error("No features were extracted. Exiting.")
         sys.exit(1)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    log.info(f"Total features extracted: {all_features.shape}")

    # --- Compute Mean Centers ---
    log.info("Computing mean center (z_V) for each base class...")
    mean_centers = {}
    for class_idx in sorted(base_dataset.class_to_idx.values()):
        class_name = base_dataset.idx_to_class[class_idx] # Get original config class name
        indices = np.where(all_labels == class_idx)[0]
        if len(indices) == 0:
            log.warning(f"No features found for class {class_name} (Index {class_idx}). Skipping.")
            continue
        class_features = all_features[indices]
        center = np.mean(class_features, axis=0)
        # The center itself is usually NOT normalized when used as target for SemAlign training (matching SemFew)
        mean_centers[class_name] = torch.from_numpy(center).float() # Store as tensor, use original name as key
        log.info(f"Computed center for {class_name} (Index {class_idx}), shape: {center.shape}")

    # --- Save Centers ---
    center_file_path = dynamic_paths['base_centers_path']
    output_dir = os.path.dirname(center_file_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Save using original config class names as keys
        torch.save({'center_mean': mean_centers, 'config': config}, center_file_path)
        log.info(f"Mean base class centers saved to: {center_file_path}")
    except Exception as e:
        log.error(f"Failed to save centers: {e}", exc_info=True)

    log.info("Center computation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mean Visual Centers (z_V) for Base Classes")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)
