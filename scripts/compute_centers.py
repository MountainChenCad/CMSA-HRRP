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
import open_clip # Assuming RemoteCLIP

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from logger import loggers
from utils import set_seed, normalize # Need normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(config):
    """Computes mean visual features for base classes using a pre-trained adapter."""

    log = loggers(os.path.join(config['paths'].get('logs', './logs'), 'compute_centers'))
    log.info("Starting base class center computation...")
    log.info(f"Config: {config}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Base Dataset ---
    log.info("Loading base dataset...")
    try:
        base_dataset = HRRPDataset(
            root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
            split='base',
            classes=config['data']['base_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude' # Match adapter input expectation
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}")
         sys.exit(1)

    # Use a simple loader to iterate through all base data
    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['alignment']['batch_size'], # Reuse batch size
        shuffle=False, # No need to shuffle
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load VLM Visual Encoder ---
    fm_config = config['model']['foundation_model']
    log.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            fm_variant = fm_config['variant']
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"
            local_weights_path = os.path.join(base_checkpoint_dir, 'foundation_models', weights_filename)
            if not os.path.exists(local_weights_path):
                 log.error(f"VLM weights file not found: {local_weights_path}")
                 sys.exit(1)
            log.info(f"Loading VLM weights from: {local_weights_path}")

            vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_variant, pretrained=local_weights_path)
            visual_encoder = vlm_model.visual.to(device).eval() # f_V
            for param in visual_encoder.parameters():
                param.requires_grad = False
            log.info("VLM Visual Encoder loaded and frozen.")
            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
            if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
        else:
            log.error(f"Unsupported foundation model: {fm_config['name']}")
            sys.exit(1)
    except Exception as e:
        log.error(f"Failed to load VLM: {e}")
        sys.exit(1)

    # --- Load Pre-trained Adapter ---
    adapter_checkpoint_path = os.path.join(config['paths']['checkpoints'], 'hrrp_adapter', 'latest.pth') # Assuming stage 1 saved here
    log.info(f"Loading pre-trained adapter from: {adapter_checkpoint_path}")
    try:
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'],
            input_channels=1, output_channels=3, output_size=expected_img_size
        ).to(device)
        adapter_checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint['adapter_state_dict'])
        adapter.eval()
        log.info("Pre-trained adapter loaded and frozen.")
    except FileNotFoundError:
        log.error(f"Adapter checkpoint file not found at {adapter_checkpoint_path}. Run adapter training first.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error loading adapter checkpoint: {e}")
        sys.exit(1)

    # --- Feature Extraction ---
    log.info("Extracting visual features for all base samples...")
    all_features = []
    all_labels = []
    with torch.no_grad():
        for hrrp_samples, labels in tqdm(base_loader, desc="Extracting Features"):
            hrrp_samples = hrrp_samples.to(device)
            try:
                pseudo_images = adapter(hrrp_samples)
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)

                visual_features = visual_encoder(pseudo_images)
                if isinstance(visual_features, tuple): visual_features = visual_features[0]
                if visual_features.dim() == 3 and visual_features.shape[1] > 1:
                     visual_features = visual_features[:, 0] # CLS token

                # Don't normalize here, normalize the final center later if needed
                all_features.append(visual_features.cpu())
                all_labels.append(labels.cpu())
            except Exception as e:
                 log.error(f"Error during feature extraction: {e}. Skipping batch.")
                 continue

    if not all_features:
         log.error("No features were extracted. Exiting.")
         sys.exit(1)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    log.info(f"Total features extracted: {all_features.shape}")

    # --- Compute Mean Centers ---
    log.info("Computing mean center for each base class...")
    mean_centers = {}
    for class_idx in sorted(base_dataset.class_to_idx.values()):
        class_name = base_dataset.idx_to_class[class_idx]
        indices = np.where(all_labels == class_idx)[0]
        if len(indices) == 0:
            log.warning(f"No features found for class {class_name} (Index {class_idx}). Skipping.")
            continue
        class_features = all_features[indices]
        center = np.mean(class_features, axis=0)
        # Optionally normalize the center
        # center = center / (np.linalg.norm(center) + 1e-8)
        mean_centers[class_name] = torch.from_numpy(center).float() # Store as tensor
        log.info(f"Computed center for {class_name} (Index {class_idx}), shape: {center.shape}")

    # --- Save Centers ---
    center_file_path = config.get('paths', {}).get('base_centers_path', './checkpoints/base_centers_mean.pth') # Add path to config
    output_dir = os.path.dirname(center_file_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        torch.save({'center_mean': mean_centers}, center_file_path)
        log.info(f"Mean base class centers saved to: {center_file_path}")
    except Exception as e:
        log.error(f"Failed to save centers: {e}")

    log.info("Center computation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mean Visual Centers for Base Classes")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    # Add the center file path to config dict if not present, using a default
    if 'paths' not in configuration: configuration['paths'] = {}
    if 'base_centers_path' not in configuration['paths']:
        configuration['paths']['base_centers_path'] = './checkpoints/base_centers_mean.pth'

    main(configuration)