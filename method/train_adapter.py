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
from utils import set_seed, normalize, load_config, get_dynamic_paths, calculate_infonce_loss # Import helpers

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

    log = loggers(os.path.join(log_dir, 'train_adapter_log.txt'))
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

    # --- Loss Configuration ---
    adapter_loss_cfg = config['training'].get('adapter_loss', {})
    alignment_loss_type = adapter_loss_cfg.get('alignment_loss_type', 'cosine').lower()
    use_contrastive = adapter_loss_cfg.get('use_contrastive', False)
    lambda_v2v = adapter_loss_cfg.get('lambda_v2v', 0.5)
    temperature_v = adapter_loss_cfg.get('temperature_v', 0.1)

    log.info(f"Alignment Loss Type: {alignment_loss_type}")
    log.info(f"Use Visual Contrastive Loss (Lcl_v2v): {use_contrastive}")
    if use_contrastive:
        log.info(f"  Lambda_v2v: {lambda_v2v}")
        log.info(f"  Temperature_v: {temperature_v}")

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
        drop_last=True # Important for contrastive loss if batch size varies
    )
    log.info(f"Base dataset loaded with {len(base_dataset)} samples.")

    # --- Load Semantic Features ---
    semantic_feature_path = dynamic_paths['semantic_features']
    log.info(f"Loading semantic features from: {semantic_feature_path}")
    try:
        if not os.path.exists(semantic_feature_path): log.error(f"Semantic features file not found: {semantic_feature_path}. Run generate_semantics.py first."); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        semantic_features = {k: normalize(v) for k,v in semantic_features.items()}
        log.info(f"Loaded and normalized semantic features for {len(semantic_features)} classes.")
    except Exception as e: log.error(f"Error loading semantic features: {e}", exc_info=True); sys.exit(1)

    # --- Load VLM (Visual Encoder - Frozen) ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path): log.error(f"VLM weights file not found: {vlm_weights_path}"); sys.exit(1)
        log.info(f"Loading VLM weights from: {vlm_weights_path}")
        vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        visual_encoder = vlm_model.visual.to(device).eval()
        for param in visual_encoder.parameters(): param.requires_grad = False
        log.info("Visual Encoder loaded and frozen.")
        expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                           (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
        if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
        log.info(f"VLM Visual Encoder expects input size: {expected_img_size}x{expected_img_size}")
    except Exception as e: log.error(f"Failed to load VLM: {e}", exc_info=True); sys.exit(1)

    # --- Adapter Initialization ---
    adapter_config = config['model'].get('adapter_1d_to_2d', {})
    hrrp_adapter = HRPPtoPseudoImage(
        hrrp_length=config['data']['target_length'], input_channels=1, output_channels=3,
        output_size=expected_img_size,
        intermediate_dim=adapter_config.get('intermediate_dim', 2048),
        activation=adapter_config.get('activation', 'relu')
    ).to(device)
    log.info(f"Initialized HRRP Adapter: {hrrp_adapter}")

    # --- Optimizer ---
    params_to_optimize = list(hrrp_adapter.parameters())
    opt_name = config['training']['optimizer'].lower()
    if opt_name == 'adam': optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    elif opt_name == 'adamw': optimizer = optim.AdamW(params_to_optimize, lr=config['training']['lr'])
    else: log.warning(f"Unsupported optimizer: {opt_name}. Using Adam."); optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])

    # --- Alignment Loss Function ---
    if alignment_loss_type == 'cosine':
        alignment_loss_fn = lambda z_v_norm, z_t_norm: 1.0 - F.cosine_similarity(z_v_norm, z_t_norm).mean()
    elif alignment_loss_type == 'l1': alignment_loss_fn = nn.L1Loss()
    elif alignment_loss_type == 'mse': alignment_loss_fn = nn.MSELoss()
    else: log.warning(f"Unsupported alignment_loss_type '{alignment_loss_type}'. Using cosine."); alignment_loss_fn = lambda z_v_norm, z_t_norm: 1.0 - F.cosine_similarity(z_v_norm, z_t_norm).mean()

    # --- LR Scheduler (Optional) ---
    scheduler = None # Add scheduler setup if needed

    # --- Training Loop ---
    log.info("Starting training loop...")
    best_loss = float('inf')
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        hrrp_adapter.train()
        total_align_loss = 0.0
        total_contrast_loss = 0.0
        total_combined_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device) # (B, 1, L)
            labels = labels.to(device)

            # Get corresponding semantic features
            try:
                target_semantics_norm = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
            except KeyError as e: log.warning(f"KeyError accessing semantic feature for label index {e}. Skipping batch {step}."); continue
            except Exception as e: log.error(f"Error getting semantic features: {e}. Skipping batch {step}.", exc_info=True); continue

            # Forward pass through adapter and frozen visual encoder
            optimizer.zero_grad()
            try:
                pseudo_images = hrrp_adapter(hrrp_samples) # (B, 3, H, W)
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                visual_features = visual_encoder(pseudo_images) # z_V
                if isinstance(visual_features, tuple): visual_features = visual_features[0]
                if visual_features.dim() == 3 and visual_features.shape[1] > 1: visual_features = visual_features[:, 0]
                elif visual_features.dim() != 2: visual_features = visual_features.view(visual_features.size(0), -1)

                expected_vis_dim = config['model']['foundation_model']['visual_encoder_dim']
                if visual_features.shape[-1] != expected_vis_dim: log.warning(f"VLM output visual dim {visual_features.shape[-1]} != config dim {expected_vis_dim}."); # Continue, loss check below

                visual_features_norm = normalize(visual_features)

            except Exception as e: log.error(f"Error during forward pass: {e}. Skipping batch {step}.", exc_info=True); continue

            # --- Loss Calculation ---
            try:
                # 1. Alignment Loss
                if visual_features_norm.shape[-1] != target_semantics_norm.shape[-1]: log.error(f"Dim mismatch! Visual: {visual_features_norm.shape}, Semantic: {target_semantics_norm.shape}. Skipping batch."); continue
                align_loss = alignment_loss_fn(visual_features_norm, target_semantics_norm)

                # 2. Visual Contrastive Loss (Lcl_v2v)
                contrastive_loss_v2v = torch.tensor(0.0, device=device)
                if use_contrastive:
                    contrastive_loss_v2v = calculate_infonce_loss(visual_features_norm, labels, temperature=temperature_v)

                # 3. Combined Loss
                combined_loss = align_loss + lambda_v2v * contrastive_loss_v2v if use_contrastive else align_loss

                # Backward pass and optimization
                combined_loss.backward()
                optimizer.step()

                total_align_loss += align_loss.item()
                total_contrast_loss += contrastive_loss_v2v.item()
                total_combined_loss += combined_loss.item()
                num_batches += 1
                postfix_dict = {'AlignL': f"{align_loss.item():.4f}"}
                if use_contrastive: postfix_dict['ContraL'] = f"{contrastive_loss_v2v.item():.4f}"
                pbar.set_postfix(postfix_dict)

            except Exception as e: log.error(f"Error during loss/backward: {e}. Skipping step {step}", exc_info=True); optimizer.zero_grad(); continue


        avg_align_loss = total_align_loss / num_batches if num_batches > 0 else 0
        avg_contrast_loss = total_contrast_loss / num_batches if num_batches > 0 else 0
        avg_combined_loss = total_combined_loss / num_batches if num_batches > 0 else 0

        log_msg = f"[Epoch {epoch+1}/{epochs}] Avg Combined Loss: {avg_combined_loss:.4f} (Align: {avg_align_loss:.4f}"
        if use_contrastive: log_msg += f", Contrast: {avg_contrast_loss:.4f})"
        else: log_msg += ")"
        log.info(log_msg)

        writer.add_scalar('Loss/Combined', avg_combined_loss, epoch)
        writer.add_scalar('Loss/Alignment', avg_align_loss, epoch)
        if use_contrastive: writer.add_scalar('Loss/Contrastive_v2v', avg_contrast_loss, epoch)

        if scheduler:
             current_lr = scheduler.get_last_lr()[0]; writer.add_scalar('LR', current_lr, epoch); log.info(f"Current LR: {current_lr:.6f}"); scheduler.step()
        else: current_lr = optimizer.param_groups[0]['lr']; writer.add_scalar('LR', current_lr, epoch)


        # --- Save Checkpoint ---
        # Use combined loss for selecting best model
        is_best = avg_combined_loss < best_loss
        if is_best: best_loss = avg_combined_loss; log.info(f"New best combined loss: {best_loss:.4f}")

        latest_ckpt_path = dynamic_paths['adapter_latest_ckpt']
        save_dict = {
            'epoch': epoch + 1, 'adapter_state_dict': hrrp_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_combined_loss,
            'align_loss': avg_align_loss, 'contrast_loss': avg_contrast_loss,
            'config': config
        }
        torch.save(save_dict, latest_ckpt_path)

        if is_best:
            best_ckpt_path = dynamic_paths['adapter_best_ckpt']
            save_dict['best_loss'] = best_loss # Add best loss marker
            torch.save(save_dict, best_ckpt_path)
            log.info(f"Best model checkpoint saved to {best_ckpt_path}")


    writer.close()
    log.info("HRRP Adapter training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRRP 1D-to-2D Adapter")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)