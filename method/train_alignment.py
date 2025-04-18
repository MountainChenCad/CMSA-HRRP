import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from model.hrrp_encoder import HRRPEncoder # Assuming this exists
from method.alignment import AlignmentModule, FusionModule
from logger import loggers
from utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config):
    """Main training function for HRRP Encoder and Alignment Module."""

    # --- Setup ---
    log_dir = os.path.join(config['paths']['logs'], 'alignment_training')
    ckpt_dir = os.path.join(config['paths']['checkpoints'], 'alignment_module')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_alignment'))
    log.info(f"Loaded configuration: {config}")
    writer = SummaryWriter(log_dir)

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading ---
    log.info("Loading base dataset...")
    base_dataset = HRRPDataset(
        root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
        split='base',
        classes=config['data']['base_classes'],
        target_length=config['data']['target_length'],
        normalization=config['data']['normalization'],
        phase_info='magnitude' # Assuming magnitude for now, adjust if needed
    )
    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['alignment']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load Semantic Features ---
    log.info(f"Loading semantic features from: {config['semantics']['feature_path']}")
    try:
        semantic_data = torch.load(config['semantics']['feature_path'], map_location='cpu')
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        log.info(f"Loaded semantic features for {len(semantic_features)} classes.")
    except FileNotFoundError:
        log.error(f"Semantic features file not found at {config['semantics']['feature_path']}")
        sys.exit(1)
    except KeyError:
         log.error(f"Key 'semantic_feature' not found in {config['semantics']['feature_path']}")
         sys.exit(1)


    # --- Model Initialization ---
    # HRRP Encoder
    hrrp_encoder = HRRPEncoder(output_dim=config['model']['hrrp_encoder']['output_dim']).to(device)

    # Alignment Module
    alignment_module = AlignmentModule(
        hrrp_feat_dim=config['model']['hrrp_encoder']['output_dim'],
        semantic_dim=config['model']['foundation_model']['text_encoder_dim'],
        hidden_dim=config['model']['alignment_module']['hidden_dim'],
        # drop=config['training']['alignment'].get('dropout', 0.1) # Add dropout if needed
    ).to(device)

    # Fusion Module (also trained here for reconstruction)
    fusion_module = FusionModule(
        aligned_hrrp_dim=config['model']['foundation_model']['text_encoder_dim'],
        semantic_dim=config['model']['foundation_model']['text_encoder_dim'],
        hidden_dim=config['model']['fusion_module']['hidden_dim'],
        output_dim=config['model']['foundation_model']['text_encoder_dim'], # Output aligns with semantic target
        # drop=config['training']['alignment'].get('dropout', 0.2)
    ).to(device)

    # --- Optimizer and Loss ---
    # Combine parameters from all modules to be trained
    params_to_optimize = list(hrrp_encoder.parameters()) + \
                         list(alignment_module.parameters()) + \
                         list(fusion_module.parameters())

    if config['training']['alignment']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['alignment']['lr'])
    # Add other optimizers like AdamW if needed
    else:
        log.warning(f"Unsupported optimizer: {config['training']['alignment']['optimizer']}. Using Adam.")
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['alignment']['lr'])

    # Define loss functions
    if config['training']['alignment']['loss_type'] == 'cosine':
        # CosineEmbeddingLoss expects input1, input2, target (1 for similar, -1 for dissimilar)
        alignment_loss_fn = nn.CosineEmbeddingLoss()
        alignment_target = torch.ones(config['training']['alignment']['batch_size']).to(device)
    elif config['training']['alignment']['loss_type'] == 'l1':
        alignment_loss_fn = nn.L1Loss()
    elif config['training']['alignment']['loss_type'] == 'mse':
        alignment_loss_fn = nn.MSELoss()
    else:
        log.warning(f"Unsupported alignment loss: {config['training']['alignment']['loss_type']}. Using L1Loss.")
        alignment_loss_fn = nn.L1Loss()

    # Reconstruction loss for the fusion module (always L1 like SemFew)
    recon_loss_fn = nn.L1Loss()

    # Learning rate scheduler (optional)
    scheduler = None
    if 'scheduler' in config['training']['alignment']:
        if config['training']['alignment']['scheduler']['type'].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['alignment']['scheduler']['step_size'],
                gamma=config['training']['alignment']['scheduler']['gamma']
            )
        # Add other schedulers if needed

    # --- Training Loop ---
    log.info("Starting alignment training...")
    for epoch in range(config['training']['alignment']['epochs']):
        hrrp_encoder.train()
        alignment_module.train()
        fusion_module.train()

        total_align_loss = 0.0
        total_recon_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{config['training']['alignment']['epochs']}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device)
            labels = labels.to(device) # Keep labels on device for efficient lookup

            # Get corresponding semantic features
            try:
                target_semantics = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
            except KeyError as e:
                log.warning(f"KeyError accessing semantic feature for label {e}. Skipping batch {step}.")
                continue

            # Forward pass
            optimizer.zero_grad()

            hrrp_features = hrrp_encoder(hrrp_samples)       # z_H = f_H(x_H)
            aligned_features = alignment_module(hrrp_features) # z'_H = h_A(z_H)

            # Alignment Loss Calculation
            if config['training']['alignment']['loss_type'] == 'cosine':
                align_loss = alignment_loss_fn(aligned_features, target_semantics, alignment_target)
            else: # L1 or MSE
                align_loss = alignment_loss_fn(aligned_features, target_semantics)

            # Reconstruction Loss Calculation (for Fusion Module)
            # Concatenate aligned features and target semantics
            fusion_input = torch.cat((aligned_features, target_semantics), dim=-1)
            reconstructed_output = fusion_module(aligned_features, target_semantics) # Pass separately for clarity inside module
            recon_loss = recon_loss_fn(reconstructed_output, target_semantics) # Target is the semantic feature itself

            # Total Loss
            loss = align_loss + recon_loss # Simple sum, can add weighting factor if needed

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_align_loss += align_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'Align Loss': f"{align_loss.item():.4f}",
                'Recon Loss': f"{recon_loss.item():.4f}",
                'Total Loss': f"{loss.item():.4f}"
            })

        avg_align_loss = total_align_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{config['training']['alignment']['epochs']}] Avg Align Loss: {avg_align_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}")
        writer.add_scalar('Loss/Alignment', avg_align_loss, epoch)
        writer.add_scalar('Loss/Reconstruction', avg_recon_loss, epoch)
        writer.add_scalar('Loss/Total', avg_total_loss, epoch)
        if scheduler:
             writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        if scheduler:
            scheduler.step()

        # --- Save Checkpoint ---
        if (epoch + 1) % 10 == 0 or epoch == config['training']['alignment']['epochs'] - 1: # Save every 10 epochs and at the end
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'hrrp_encoder_state_dict': hrrp_encoder.state_dict(),
                'alignment_module_state_dict': alignment_module.state_dict(),
                'fusion_module_state_dict': fusion_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config # Store config for reference
            }, ckpt_path)
            log.info(f"Checkpoint saved to {ckpt_path}")

    # Save the final model also as 'latest.pth'
    latest_ckpt_path = os.path.join(ckpt_dir, 'latest.pth')
    torch.save({
        'epoch': config['training']['alignment']['epochs'],
        'hrrp_encoder_state_dict': hrrp_encoder.state_dict(),
        'alignment_module_state_dict': alignment_module.state_dict(),
        'fusion_module_state_dict': fusion_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, latest_ckpt_path)
    log.info(f"Final model saved to {latest_ckpt_path}")

    writer.close()
    log.info("Alignment training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRRP Encoder and Alignment Module")
    parser.add_argument('--config', type=str, default='hrrp_fsl_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    main(configuration)
