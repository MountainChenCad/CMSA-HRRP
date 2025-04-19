# scripts/compute_centers_1dcnn.py
import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from model.hrrp_encoder import HRRPEncoder # Import the 1D CNN
from logger import loggers
from utils import set_seed, load_config, get_dynamic_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str):
    """Computes mean features (z_H) for base classes using a pre-trained 1D CNN."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup Logging ---
    log_dir = dynamic_paths['baseline_centers_1dcnn_log_dir']
    os.makedirs(log_dir, exist_ok=True)
    log = loggers(os.path.join(log_dir, 'compute_centers_1dcnn_log.txt'))
    log.info("Starting baseline 1D CNN center computation (z_H)...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"Centers will be saved to: {dynamic_paths['baseline_centers_1dcnn_path']}")
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
            phase_info='magnitude'
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}", exc_info=True)
         sys.exit(1)

    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load Pre-trained 1D CNN Encoder ---
    encoder_checkpoint_path = dynamic_paths['baseline_cnn_latest_ckpt'] # Or use best
    log.info(f"Loading pre-trained 1D CNN encoder from: {encoder_checkpoint_path}")
    try:
        if not os.path.exists(encoder_checkpoint_path):
            log.error(f"Encoder checkpoint not found: {encoder_checkpoint_path}"); sys.exit(1)

        cnn_config = config.get('baseline_cnn', config['model'].get('hrrp_encoder', {}))
        feature_dim = cnn_config.get('output_dim', 512)
        encoder = HRRPEncoder(
            input_channels=1, output_dim=feature_dim,
            layers=cnn_config.get('layers', [64, 128, 256, 512]),
            kernel_size=cnn_config.get('kernel_size', 7)
        ).to(device)
        checkpoint_data = torch.load(encoder_checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
        encoder.eval()
        log.info(f"Baseline 1D CNN encoder loaded (from epoch {checkpoint_data.get('epoch', 'N/A')}). Feature Dim: {feature_dim}")
    except Exception as e:
        log.error(f"Error loading baseline CNN encoder: {e}", exc_info=True); sys.exit(1)

    # --- Feature Extraction ---
    log.info("Extracting features (z_H) for all base samples...")
    all_features = []
    all_labels = []
    with torch.no_grad():
        for hrrp_samples, labels in tqdm(base_loader, desc="Extracting Features (1D CNN)"):
            hrrp_samples = hrrp_samples.to(device)
            try:
                features = encoder(hrrp_samples) # z_H
                # Store UNNORMALIZED features for computing the mean center
                all_features.append(features.cpu())
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
    log.info("Computing mean center (z_H) for each base class...")
    mean_centers = {}
    for class_idx in sorted(base_dataset.class_to_idx.values()):
        class_name = base_dataset.idx_to_class[class_idx]
        indices = np.where(all_labels == class_idx)[0]
        if len(indices) == 0:
            log.warning(f"No features found for class {class_name} (Index {class_idx}). Skipping.")
            continue
        class_features = all_features[indices]
        center = np.mean(class_features, axis=0)
        # Store UNNORMALIZED centers
        mean_centers[class_name] = torch.from_numpy(center).float()
        log.info(f"Computed center for {class_name} (Index {class_idx}), shape: {center.shape}")

    # --- Save Centers ---
    center_file_path = dynamic_paths['baseline_centers_1dcnn_path']
    output_dir = os.path.dirname(center_file_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        torch.save({'center_mean_1dcnn': mean_centers, 'config': config}, center_file_path)
        log.info(f"Mean base class 1D CNN centers (z_H) saved to: {center_file_path}")
    except Exception as e:
        log.error(f"Failed to save centers: {e}", exc_info=True)

    log.info("1D CNN center computation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mean 1D CNN Features (z_H) for Base Classes")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)