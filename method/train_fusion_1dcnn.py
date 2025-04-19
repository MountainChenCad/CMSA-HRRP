# method/train_fusion_1dcnn.py
import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from model.hrrp_encoder import HRRPEncoder # Baseline encoder
from method.alignment import SemAlignModule # Reusable fusion module
from logger import loggers
from utils import set_seed, normalize, load_config, get_dynamic_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str):
    """Trains the SemAlign/Fusion module for baseline 1D CNN features (z_H)."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup ---
    log_dir = dynamic_paths['baseline_fusion_log_dir']
    ckpt_dir = dynamic_paths['baseline_fusion_checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_fusion_1dcnn_log.txt'))
    log.info("Starting Baseline Fusion Module training (for z_H)...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"Text Type used for fusion: {config['semantics']['generation']['text_type']}")
    log.info(f"Checkpoints will be saved to: {ckpt_dir}")
    log.info(f"Logs will be saved to: {log_dir}")
    writer = SummaryWriter(log_dir)

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading (Base Set) ---
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
         log.error(f"Failed to initialize Base HRRPDataset: {e}", exc_info=True); sys.exit(1)

    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
        drop_last=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load Semantic Features (z_T) ---
    # Use semantics corresponding to the VLM variant specified in config, even for baseline fusion
    semantic_feature_path = dynamic_paths['semantic_features']
    log.info(f"Loading semantic features (z_T) from: {semantic_feature_path}")
    try:
        if not os.path.exists(semantic_feature_path):
             log.error(f"Semantic features file not found: {semantic_feature_path}"); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        # Use text_encoder_dim from VLM config, as z_T comes from VLM
        semantic_dim = config['model']['foundation_model']['text_encoder_dim']
        loaded_sem_dim = list(semantic_features.values())[0].shape[-1]
        if loaded_sem_dim != semantic_dim:
             log.warning(f"Loaded semantic dim {loaded_sem_dim} != VLM config dim {semantic_dim}")
             semantic_dim = loaded_sem_dim
        log.info(f"Loaded semantic features for {len(semantic_features)} classes. Dim: {semantic_dim}")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-computed Base Centers (z_H) ---
    center_file_path = dynamic_paths['baseline_centers_1dcnn_path']
    log.info(f"Loading pre-computed base centers (z_H) from: {center_file_path}")
    try:
        if not os.path.exists(center_file_path):
            log.error(f"Baseline centers file not found: {center_file_path}. Run compute_centers_1dcnn.py first."); sys.exit(1)
        center_data = torch.load(center_file_path, map_location='cpu')
        base_centers = {k: v.float().to(device) for k, v in center_data['center_mean_1dcnn'].items()}
        # Infer feature dim from centers
        feature_dim = list(base_centers.values())[0].shape[0]
        log.info(f"Loaded 1D CNN base centers for {len(base_centers)} classes. Dim (z_H): {feature_dim}")
    except Exception as e:
        log.error(f"Error loading 1D CNN base centers: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained 1D CNN Encoder (Frozen) ---
    encoder_checkpoint_path = dynamic_paths['baseline_cnn_latest_ckpt'] # Or use best
    log.info(f"Loading pre-trained 1D CNN encoder from: {encoder_checkpoint_path}")
    try:
        if not os.path.exists(encoder_checkpoint_path):
             log.error(f"Encoder checkpoint not found: {encoder_checkpoint_path}"); sys.exit(1)
        cnn_config = config.get('baseline_cnn', config['model'].get('hrrp_encoder', {}))
        # Ensure encoder output dim matches loaded centers dim
        if cnn_config.get('output_dim', 512) != feature_dim:
             log.warning(f"Configured CNN output dim {cnn_config.get('output_dim', 512)} != loaded center dim {feature_dim}. Using center dim.")
        encoder = HRRPEncoder(
            input_channels=1, output_dim=feature_dim, # Use inferred dim
            layers=cnn_config.get('layers', [64, 128, 256, 512]),
            kernel_size=cnn_config.get('kernel_size', 7)
        ).to(device)
        checkpoint_data = torch.load(encoder_checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
        encoder.eval()
        for param in encoder.parameters(): param.requires_grad = False # Freeze encoder
        log.info(f"Baseline 1D CNN encoder loaded and frozen (from epoch {checkpoint_data.get('epoch', 'N/A')}).")
    except Exception as e:
        log.error(f"Error loading baseline CNN encoder: {e}", exc_info=True); sys.exit(1)

    # --- Initialize SemAlign Module (Trainable) ---
    # Input: z_H + z_T, Output: z_H
    fusion_module = SemAlignModule(
        visual_dim=feature_dim, # Input visual dim is z_H dim
        semantic_dim=semantic_dim, # Input semantic dim is z_T dim
        hidden_dim=config['model']['fusion_module']['hidden_dim'],
        output_dim=feature_dim, # Output should match z_H dim
        drop=config['training'].get('dropout_semalign', 0.2)
    ).to(device)
    log.info(f"Initialized Fusion Module for 1D CNN: {fusion_module}")

    # --- Optimizer ---
    params_to_optimize = list(fusion_module.parameters())
    opt_name = config['training']['optimizer'].lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    elif opt_name == 'adamw':
         optimizer = optim.AdamW(params_to_optimize, lr=config['training']['lr'])
    else:
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])

    # --- Loss function: L1 distance ---
    recon_loss_fn = nn.L1Loss()
    log.info("Using L1 reconstruction loss.")

    # --- LR scheduler (optional) ---
    scheduler = None

    # --- Training Loop ---
    log.info("Starting baseline fusion module training loop...")
    best_loss = float('inf')
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        fusion_module.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device)
            labels = labels.to(device)

            # Get corresponding semantic features (z_T) and target centers (z_H)
            try:
                target_semantics = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
                target_centers = torch.stack([base_centers[base_dataset.idx_to_class[l.item()]] for l in labels])
            except KeyError as e:
                log.warning(f"KeyError accessing semantic/center for label index {e}. Skipping batch {step}.")
                continue
            except Exception as e:
                 log.error(f"Error getting features/centers: {e}. Skipping batch {step}.", exc_info=True)
                 continue

            # Forward pass through frozen 1D CNN encoder to get z_H
            with torch.no_grad():
                try:
                    cnn_features = encoder(hrrp_samples) # z_H
                    # Input to fusion module (cnn_features) should NOT be normalized
                except Exception as e:
                     log.error(f"Error during 1D CNN forward pass: {e}. Skipping batch {step}.", exc_info=True)
                     continue

            # Forward pass through Fusion module (trainable)
            optimizer.zero_grad()
            try:
                 # Check dims
                 if cnn_features.shape[-1] != feature_dim: continue
                 if target_semantics.shape[-1] != semantic_dim: continue

                 reconstructed_centers = fusion_module(cnn_features, target_semantics) # r = h([z_H, z_T])

                 # Reconstruction Loss
                 if reconstructed_centers.shape != target_centers.shape: continue
                 loss = recon_loss_fn(reconstructed_centers, target_centers) # L1 distance

                 # Backward pass and optimization
                 loss.backward()
                 optimizer.step()

                 total_loss += loss.item()
                 num_batches += 1
                 pbar.set_postfix({'Recon Loss': f"{loss.item():.4f}"})

            except Exception as e:
                log.error(f"Error during fusion forward/loss/backward: {e}. Skipping step {step}", exc_info=True)
                optimizer.zero_grad()
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{epochs}] Avg Recon Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Reconstruction_1DCNN', avg_loss, epoch)
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('LR_1DCNN_Fusion', current_lr, epoch)
            log.info(f"Current LR: {current_lr:.6f}")
            scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR_1DCNN_Fusion', current_lr, epoch)

        # --- Save Checkpoint ---
        is_best = avg_loss < best_loss
        if is_best:
             best_loss = avg_loss
             log.info(f"New best reconstruction loss: {best_loss:.4f}")

        # Save latest checkpoint
        latest_ckpt_path = dynamic_paths['baseline_fusion_latest_ckpt']
        torch.save({
            'epoch': epoch + 1,
            'fusion_1dcnn_state_dict': fusion_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, latest_ckpt_path)

        # Save best checkpoint
        if is_best:
            best_ckpt_path = dynamic_paths['baseline_fusion_best_ckpt']
            torch.save({
                'epoch': epoch + 1,
                'fusion_1dcnn_state_dict': fusion_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, best_ckpt_path)
            log.info(f"Best baseline fusion model saved to {best_ckpt_path}")

    writer.close()
    log.info("Baseline Fusion Module training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fusion Module for Baseline 1D CNN")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)