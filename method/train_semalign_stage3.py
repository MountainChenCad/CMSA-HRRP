# method/train_semalign_stage3.py
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
from method.alignment import SemAlignModule # Use the renamed module
from logger import loggers
from utils import set_seed, normalize, load_config, get_dynamic_paths # Import helpers

logging.basicConfig(level=logging.INFO) # Basic config for early messages
logger = logging.getLogger(__name__)

def main(config_path: str):
    """Main training function for SemAlign Module (h_F) for visual features (z_V)."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup ---
    log_dir = dynamic_paths['semalign_log_dir']
    ckpt_dir = dynamic_paths['semalign_checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_semalign_log.txt'))
    log.info("Starting SemAlign Module training (Stage 3)...")
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
             log.error(f"Semantic features file not found: {semantic_feature_path}"); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        semantic_dim = config['model']['foundation_model']['text_encoder_dim']
        loaded_sem_dim = list(semantic_features.values())[0].shape[-1]
        if loaded_sem_dim != semantic_dim:
             log.warning(f"Loaded semantic dim {loaded_sem_dim} != config dim {semantic_dim}")
             semantic_dim = loaded_sem_dim
        log.info(f"Loaded semantic features for {len(semantic_features)} classes. Dim: {semantic_dim}")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Pre-computed Base Centers (z_V) ---
    center_file_path = dynamic_paths['base_centers_path']
    log.info(f"Loading pre-computed base centers (z_V) from: {center_file_path}")
    try:
        if not os.path.exists(center_file_path):
            log.error(f"Base centers file not found at {center_file_path}. Run compute_centers.py first for this config.")
            sys.exit(1)
        center_data = torch.load(center_file_path, map_location='cpu')
        # Centers dict uses original config class names as keys
        base_centers = {k: v.float().to(device) for k, v in center_data['center_mean'].items()}
        visual_dim = config['model']['foundation_model']['visual_encoder_dim']
        loaded_vis_dim = list(base_centers.values())[0].shape[0]
        if loaded_vis_dim != visual_dim:
             log.warning(f"Loaded center visual dim {loaded_vis_dim} != config dim {visual_dim}")
             visual_dim = loaded_vis_dim
        log.info(f"Loaded base centers for {len(base_centers)} classes. Dim: {visual_dim}")
    except Exception as e:
        log.error(f"Error loading base centers: {e}", exc_info=True)
        sys.exit(1)

    # --- Load VLM Visual Encoder (Frozen) ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path): log.error(f"VLM weights not found: {vlm_weights_path}"); sys.exit(1)
        log.info(f"Loading VLM weights from: {vlm_weights_path}")

        vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        visual_encoder = vlm_model.visual.to(device).eval() # f_V
        for param in visual_encoder.parameters(): param.requires_grad = False
        log.info("VLM Visual Encoder loaded and frozen.")
        expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                           (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
        if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
    except Exception as e: log.error(f"Failed to load VLM: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained Adapter (Frozen) ---
    adapter_checkpoint_path = dynamic_paths['adapter_latest_ckpt'] # Or use 'adapter_best_ckpt'
    log.info(f"Loading pre-trained adapter from: {adapter_checkpoint_path}")
    try:
        if not os.path.exists(adapter_checkpoint_path): log.error(f"Adapter checkpoint not found: {adapter_checkpoint_path}"); sys.exit(1)
        adapter_config = config['model'].get('adapter_1d_to_2d', {})
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'], input_channels=1, output_channels=3, output_size=expected_img_size,
            intermediate_dim=adapter_config.get('intermediate_dim', 2048)
        ).to(device)
        adapter_checkpoint_data = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint_data['adapter_state_dict'])
        adapter.eval()
        for param in adapter.parameters(): param.requires_grad = False # Freeze adapter
        log.info(f"Pre-trained adapter loaded and frozen (from epoch {adapter_checkpoint_data.get('epoch', 'N/A')}).")
    except Exception as e: log.error(f"Error loading adapter: {e}", exc_info=True); sys.exit(1)

    # --- Initialize SemAlign Module (Trainable) ---
    semalign_module = SemAlignModule(
        visual_dim=visual_dim, # Use inferred/validated dim
        semantic_dim=semantic_dim, # Use inferred/validated dim
        hidden_dim=config['model']['fusion_module']['hidden_dim'],
        output_dim=visual_dim, # Output matches visual center dim
        drop=config['training'].get('dropout_semalign', 0.2)
    ).to(device)
    log.info(f"Initialized SemAlign Module: {semalign_module}")

    # --- Optimizer ---
    params_to_optimize = list(semalign_module.parameters())
    opt_name = config['training']['optimizer'].lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(params_to_optimize, lr=config['training']['lr'])
    else:
        log.warning(f"Unsupported optimizer: {opt_name}. Using Adam.")
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])

    # --- Loss function: L1 distance ---
    recon_loss_fn = nn.L1Loss()
    log.info("Using L1 reconstruction loss.")

    # --- LR scheduler (optional) ---
    scheduler = None # Add scheduler setup if needed

    # --- Training Loop ---
    log.info("Starting SemAlign training loop...")
    best_loss = float('inf')
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        semalign_module.train() # Set SemAlign to train mode
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device)
            labels = labels.to(device)

            # Get corresponding semantic features and target centers
            try:
                # Use original config class names (mapped by dataset) to get features/centers
                target_semantics = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
                target_centers = torch.stack([base_centers[base_dataset.idx_to_class[l.item()]] for l in labels])
            except KeyError as e:
                log.warning(f"KeyError accessing semantic/center for label index {e} (class: {base_dataset.idx_to_class.get(e.item(), 'Unknown')}). Skipping batch {step}.")
                continue
            except Exception as e:
                 log.error(f"Error getting semantic features or centers: {e}. Skipping batch {step}.", exc_info=True)
                 continue

            # Forward pass through frozen adapter and visual encoder to get z_V
            with torch.no_grad():
                try:
                    pseudo_images = adapter(hrrp_samples)
                    if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                         pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                    visual_features = visual_encoder(pseudo_images) # z_V
                    if isinstance(visual_features, tuple): visual_features = visual_features[0]
                    if visual_features.dim() == 3 and visual_features.shape[1] > 1: visual_features = visual_features[:, 0]
                    # Input to SemAlign (visual_features) should NOT be normalized here (matching SemFew)
                except Exception as e:
                     log.error(f"Error during adapter/VLM forward pass: {e}. Skipping batch {step}.", exc_info=True)
                     continue

            # Forward pass through SemAlign module (trainable)
            optimizer.zero_grad()
            try:
                 # Ensure dimensions match before input to SemAlign
                 if visual_features.shape[-1] != visual_dim:
                      log.error(f"Visual feature dim {visual_features.shape[-1]} != SemAlign input dim {visual_dim}. Skipping batch.")
                      continue
                 if target_semantics.shape[-1] != semantic_dim:
                      log.error(f"Semantic feature dim {target_semantics.shape[-1]} != SemAlign input dim {semantic_dim}. Skipping batch.")
                      continue

                 reconstructed_centers = semalign_module(visual_features, target_semantics) # r = h([z_V, z_T])

                 # Reconstruction Loss Calculation (SemFew Eq. 4)
                 # Ensure dimensions match before loss
                 if reconstructed_centers.shape != target_centers.shape:
                      log.error(f"Reconstructed center shape {reconstructed_centers.shape} != Target center shape {target_centers.shape}. Skipping loss.")
                      continue

                 loss = recon_loss_fn(reconstructed_centers, target_centers) # L1 distance

                 # Backward pass and optimization (only updates SemAlign module)
                 loss.backward()
                 optimizer.step()

                 total_loss += loss.item()
                 num_batches += 1
                 pbar.set_postfix({'Recon Loss': f"{loss.item():.4f}"})

            except Exception as e:
                log.error(f"Error during SemAlign forward/loss/backward: {e}. Skipping step {step}", exc_info=True)
                optimizer.zero_grad() # Ensure grads are cleared
                continue


        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{epochs}] Avg Recon Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Reconstruction', avg_loss, epoch)
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
             log.info(f"New best reconstruction loss: {best_loss:.4f}")

        # Save latest checkpoint
        latest_ckpt_path = dynamic_paths['semalign_latest_ckpt']
        torch.save({
            'epoch': epoch + 1,
            'semalign_state_dict': semalign_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config # Save config for reference
        }, latest_ckpt_path)

        # Save best checkpoint if current is best
        if is_best:
            best_ckpt_path = dynamic_paths['semalign_best_ckpt']
            torch.save({
                'epoch': epoch + 1,
                'semalign_state_dict': semalign_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, best_ckpt_path)
            log.info(f"Best SemAlign model saved to {best_ckpt_path}")

    writer.close()
    log.info("SemAlign Module training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SemAlign Module (Stage 3)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)