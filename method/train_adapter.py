# method/train_adapter.py
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
    """Main training function for HRRP 1D-to-2D Adapter."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup ---
    log_dir = dynamic_paths['adapter_log_dir']
    ckpt_dir = dynamic_paths['adapter_checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_adapter_log.txt')) # Use specific log file name
    log.info("Starting HRRP Adapter training...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"VLM Variant: {config['model']['foundation_model']['variant']}")
    log.info(f"Text Type: {config['semantics']['generation']['text_type']}")
    log.info(f"Checkpoints will be saved to: {ckpt_dir}")
    log.info(f"Logs will be saved to: {log_dir}")
    writer = SummaryWriter(log_dir)

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading (Base Set) ---
    log.info("Loading base dataset...")
    try:
        # Construct dataset paths
        sim_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'simulated_hrrp')
        meas_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'measured_hrrp')
        base_dataset = HRRPDataset(
            root_dirs=[sim_path, meas_path],
            split='base',
            classes=config['data']['base_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude' # Assuming adapter expects magnitude input
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}", exc_info=True)
         sys.exit(1)

    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
        drop_last=True
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load Semantic Features ---
    semantic_feature_path = dynamic_paths['semantic_features']
    log.info(f"Loading semantic features from: {semantic_feature_path}")
    try:
        if not os.path.exists(semantic_feature_path):
             log.error(f"Semantic features file not found: {semantic_feature_path}. Run generate_semantics.py first.")
             sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        # Use original config class names as keys
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        # Ensure features are normalized (should be done during generation, but double-check)
        semantic_features = {k: normalize(v) for k,v in semantic_features.items()}
        log.info(f"Loaded and normalized semantic features for {len(semantic_features)} classes.")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}", exc_info=True)
        sys.exit(1)

    # --- Load VLM (Visual Encoder - Frozen) ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path):
            log.error(f"VLM weights file not found: {vlm_weights_path}")
            sys.exit(1)
        log.info(f"Loading VLM weights from: {vlm_weights_path}")

        vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        visual_encoder = vlm_model.visual # f_V

        visual_encoder = visual_encoder.to(device).eval()

        # Freeze VLM parameters
        for param in visual_encoder.parameters():
            param.requires_grad = False
        # No need to freeze text encoder here as we only use visual encoder
        log.info("Visual Encoder loaded and frozen.")

        # Get expected image size for the visual encoder
        expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                           (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
        if isinstance(expected_img_size, (tuple, list)):
             expected_img_size = expected_img_size[0]
        log.info(f"VLM Visual Encoder expects input size: {expected_img_size}x{expected_img_size}")

    except Exception as e:
        log.error(f"Failed to load VLM: {e}", exc_info=True)
        sys.exit(1)

    # --- Adapter Initialization ---
    adapter_config = config['model'].get('adapter_1d_to_2d', {}) # Get adapter specific config
    hrrp_adapter = HRPPtoPseudoImage(
        hrrp_length=config['data']['target_length'],
        input_channels=1, # Hardcoded for magnitude
        output_channels=3, # Standard for visual models
        output_size=expected_img_size,
        intermediate_dim=adapter_config.get('intermediate_dim', 2048), # Use config value
        activation=adapter_config.get('activation', 'relu') # Use config value or default
    ).to(device)
    log.info(f"Initialized HRRP Adapter: {hrrp_adapter}")

    # --- Optimizer ---
    params_to_optimize = list(hrrp_adapter.parameters())
    opt_name = config['training']['optimizer'].lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    elif opt_name == 'adamw':
         optimizer = optim.AdamW(params_to_optimize, lr=config['training']['lr'])
    else:
        log.warning(f"Unsupported optimizer: {opt_name}. Using Adam.")
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])

    # --- Loss Function ---
    loss_type = config['training']['loss_type'].lower()
    if loss_type == 'cosine':
        # We want to maximize similarity == minimize (1 - similarity)
        alignment_loss_fn = lambda z_v_norm, z_t_norm: 1.0 - F.cosine_similarity(z_v_norm, z_t_norm).mean()
    elif loss_type == 'l1':
        alignment_loss_fn = nn.L1Loss()
    elif loss_type == 'mse':
        alignment_loss_fn = nn.MSELoss()
    else:
        log.warning(f"Unsupported loss_type '{loss_type}'. Using cosine.")
        alignment_loss_fn = lambda z_v_norm, z_t_norm: 1.0 - F.cosine_similarity(z_v_norm, z_t_norm).mean()
    log.info(f"Using alignment loss: {loss_type}")

    # --- LR Scheduler (Optional) ---
    scheduler = None # Add scheduler setup if needed based on config

    # --- Training Loop ---
    log.info("Starting training loop...")
    best_loss = float('inf')
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        hrrp_adapter.train() # Set adapter to train mode
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device) # (B, 1, L)
            labels = labels.to(device)

            # Get corresponding semantic features
            try:
                # Use original config class names (mapped by dataset) to get features
                target_semantics_norm = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
                # Features should already be normalized
            except KeyError as e:
                log.warning(f"KeyError accessing semantic feature for label index {e} (class: {base_dataset.idx_to_class.get(e.item(), 'Unknown')}). Skipping batch {step}.")
                continue
            except Exception as e:
                 log.error(f"Error getting semantic features: {e}. Skipping batch {step}.", exc_info=True)
                 continue

            # Forward pass through adapter and frozen visual encoder
            optimizer.zero_grad()
            try:
                pseudo_images = hrrp_adapter(hrrp_samples) # (B, 3, H, W)

                # Resize if adapter output doesn't exactly match VLM input (should match with MLP approach)
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     log.warning(f"Adapter output size {pseudo_images.shape[-2:]} doesn't match VLM expected size {(expected_img_size, expected_img_size)}. Resizing.")
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)

                visual_features = visual_encoder(pseudo_images) # z_V = f_V(h_1D_to_2D(x_H))

                # Handle potential tuple output or CLS token extraction for ViTs
                if isinstance(visual_features, tuple):
                     visual_features = visual_features[0]
                # Assuming ViT-like output [B, N_tokens, D] or [B, D] if pooled
                # If output is [B, N, D], take the first token (CLS)
                if visual_features.dim() == 3 and visual_features.shape[1] > 1:
                     visual_features = visual_features[:, 0]
                elif visual_features.dim() != 2:
                     # If not 2D or expected 3D, log error and attempt flatten
                     log.warning(f"Unexpected visual feature dim: {visual_features.shape}. Attempting flatten.")
                     visual_features = visual_features.view(visual_features.size(0), -1) # Flatten features

                # Check if visual feature dimension matches expected dimension from config
                expected_vis_dim = config['model']['foundation_model']['visual_encoder_dim']
                if visual_features.shape[-1] != expected_vis_dim:
                     log.warning(f"VLM output visual dim {visual_features.shape[-1]} != config dim {expected_vis_dim}. Check VLM variant/config.")
                     # Attempt linear projection if dimensions mismatch? Or just error out?
                     # For now, continue but loss calculation might fail.
                     # Add projection layer? Let's assume they match for now.

                # Normalize visual features
                visual_features_norm = normalize(visual_features)

            except Exception as e:
                 log.error(f"Error during forward pass (adapter or visual encoder): {e}. Skipping batch {step}.", exc_info=True)
                 continue

            # Alignment Loss Calculation
            try:
                 # Ensure dimensions match before loss calculation
                 if visual_features_norm.shape[-1] != target_semantics_norm.shape[-1]:
                      log.error(f"Dimension mismatch! Visual features: {visual_features_norm.shape}, Semantic features: {target_semantics_norm.shape}. Skipping loss calculation.")
                      continue

                 loss = alignment_loss_fn(visual_features_norm, target_semantics_norm)

                 # Backward pass and optimization (only updates adapter)
                 loss.backward()
                 optimizer.step()

                 total_loss += loss.item()
                 num_batches += 1
                 pbar.set_postfix({'Align Loss': f"{loss.item():.4f}"})

            except Exception as e:
                log.error(f"Error during loss calculation or backward pass: {e}. Skipping step {step}", exc_info=True)
                optimizer.zero_grad() # Ensure grads are cleared if backward failed
                continue


        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{epochs}] Avg Align Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Alignment', avg_loss, epoch)
        if scheduler:
             current_lr = scheduler.get_last_lr()[0]
             writer.add_scalar('LR', current_lr, epoch)
             log.info(f"Current LR: {current_lr:.6f}")
             scheduler.step()
        else:
             current_lr = optimizer.param_groups[0]['lr']
             writer.add_scalar('LR', current_lr, epoch)


        # --- Save Checkpoint ---
        is_best = avg_loss < best_loss
        if is_best:
             best_loss = avg_loss
             log.info(f"New best loss: {best_loss:.4f}")

        # Save latest checkpoint
        latest_ckpt_path = dynamic_paths['adapter_latest_ckpt']
        torch.save({
            'epoch': epoch + 1,
            'adapter_state_dict': hrrp_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config # Save config for reference
        }, latest_ckpt_path)

        # Save best checkpoint if current is best
        if is_best:
            best_ckpt_path = dynamic_paths['adapter_best_ckpt']
            torch.save({
                'epoch': epoch + 1,
                'adapter_state_dict': hrrp_adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, best_ckpt_path)
            log.info(f"Best model checkpoint saved to {best_ckpt_path}")


    writer.close()
    log.info("HRRP Adapter training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRRP 1D-to-2D Adapter")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)
