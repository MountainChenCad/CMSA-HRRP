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
import open_clip # Assuming RemoteCLIP

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
# from model.hrrp_encoder import HRRPEncoder # No longer needed
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage # Import the new adapter
# from method.alignment import AlignmentModule, FusionModule # No longer needed for this training
from logger import loggers
from utils import set_seed, normalize # Need normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config):
    """Main training function for HRRP 1D-to-2D Adapter."""

    # --- Setup ---
    # Modify log/ckpt dirs to reflect adapter training
    log_dir = os.path.join(config['paths'].get('logs', './logs'), 'adapter_training')
    ckpt_dir = os.path.join(config['paths'].get('checkpoints', './checkpoints'), 'hrrp_adapter') # New checkpoint dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_adapter'))
    log.info(f"Loaded configuration: {config}")
    writer = SummaryWriter(log_dir)

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading (Base Set) ---
    log.info("Loading base dataset...")
    try:
        base_dataset = HRRPDataset(
            root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
            split='base',
            classes=config['data']['base_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude' # Assuming adapter expects magnitude input (1 channel)
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}")
         sys.exit(1)

    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['alignment']['batch_size'], # Reuse batch size setting
        shuffle=True,
        num_workers=config.get('num_workers', 0),
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
    except Exception as e:
        log.error(f"Error loading semantic features: {e}")
        sys.exit(1)

    # --- Load VLM (Visual + Text Encoders) ---
    fm_config = config['model']['foundation_model']
    log.info(f"Loading Foundation Model: {fm_config['name']} ({fm_config['variant']}) for visual/text encoders")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            # Construct path dynamically
            fm_variant = fm_config['variant']
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"
            local_weights_path = os.path.join(base_checkpoint_dir, 'foundation_models', weights_filename)

            if not os.path.exists(local_weights_path):
                 log.error(f"VLM weights file not found: {local_weights_path}")
                 sys.exit(1)
            log.info(f"Loading VLM weights from: {local_weights_path}")

            vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_variant, pretrained=local_weights_path)
            visual_encoder = vlm_model.visual # f_V
            text_encoder = vlm_model # Use full model for text encoding f_T
            tokenizer = open_clip.get_tokenizer(fm_variant) # Needed if encoding text on the fly (not needed here)

            visual_encoder = visual_encoder.to(device).eval()
            text_encoder = text_encoder.to(device).eval()

            # Freeze VLM parameters
            for param in visual_encoder.parameters():
                param.requires_grad = False
            for param in text_encoder.parameters():
                 param.requires_grad = False
            log.info("Visual and Text Encoders loaded and frozen.")

            # Get expected image size for the visual encoder
            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224) # Common default
            if isinstance(expected_img_size, (tuple, list)):
                 expected_img_size = expected_img_size[0] # Assume square H=W
            log.info(f"VLM Visual Encoder expects input size: {expected_img_size}x{expected_img_size}")

        else:
            log.error(f"Unsupported foundation model: {fm_config['name']}")
            sys.exit(1)
    except Exception as e:
        log.error(f"Failed to load VLM: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

    # --- Adapter Initialization ---
    # Assuming HRRPDataset outputs (B, 1, L) for magnitude
    hrrp_adapter = HRPPtoPseudoImage(
        hrrp_length=config['data']['target_length'],
        input_channels=1, # Hardcoded for magnitude, adjust if using complex
        output_channels=3, # Standard for visual models
        output_size=expected_img_size, # Match VLM input size
        # intermediate_channels=config['model'].get('adapter_intermediate_channels', [64, 128, 256]), # Add to config if needed
        # kernel_size=config['model'].get('adapter_kernel_size', 4),
        # stride=config['model'].get('adapter_stride', 2),
        # padding=config['model'].get('adapter_padding', 1),
        # activation=config['model'].get('adapter_activation', 'relu'),
    ).to(device)

    # --- Optimizer ---
    # Optimize ONLY the adapter parameters
    params_to_optimize = list(hrrp_adapter.parameters())

    if config['training']['alignment']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['alignment']['lr'])
    else:
        log.warning(f"Unsupported optimizer: {config['training']['alignment']['optimizer']}. Using Adam.")
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['alignment']['lr'])

    # Loss function: Cosine Similarity based (like CLIP loss)
    # We want to maximize similarity == minimize (1 - similarity)
    alignment_loss_fn = lambda z_v, z_t: 1.0 - F.cosine_similarity(z_v, z_t).mean()

    # Learning rate scheduler (optional) - reuse config structure
    scheduler = None
    # ... (scheduler setup code as before) ...

    # --- Training Loop ---
    log.info("Starting HRRP Adapter training...")
    best_loss = float('inf')
    for epoch in range(config['training']['alignment']['epochs']):
        hrrp_adapter.train() # Set adapter to train mode

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{config['training']['alignment']['epochs']}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device) # (B, C_in, L)
            labels = labels.to(device)

            # Get corresponding semantic features
            try:
                target_semantics_norm = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
                # Ensure z_T is normalized (generate_semantics should have done this, but double check)
                target_semantics_norm = normalize(target_semantics_norm)
            except KeyError as e:
                log.warning(f"KeyError accessing semantic feature for label {e}. Skipping batch {step}.")
                continue
            except Exception as e:
                 log.error(f"Error getting semantic features: {e}. Skipping batch {step}.")
                 continue

            # Forward pass through adapter and frozen visual encoder
            optimizer.zero_grad()
            try:
                pseudo_images = hrrp_adapter(hrrp_samples) # (B, C_out, H, W)

                # Check shape matches VLM expectation
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     log.warning(f"Adapter output size {pseudo_images.shape[-2:]} doesn't match VLM expected size {(expected_img_size, expected_img_size)}. Resizing.")
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)

                visual_features = visual_encoder(pseudo_images) # z_V = f_V(h_1D_to_2D(x_H))
                # Handle potential [CLS] token if ViT, take appropriate output
                if isinstance(visual_features, tuple): # Some models return tuple
                     visual_features = visual_features[0]
                if visual_features.dim() == 3 and visual_features.shape[1] > 1: # e.g., ViT output with tokens
                     visual_features = visual_features[:, 0] # Take CLS token

                # Normalize visual features
                visual_features_norm = normalize(visual_features)

            except Exception as e:
                 log.error(f"Error during forward pass (adapter or visual encoder): {e}. Skipping batch {step}.")
                 import traceback
                 log.error(traceback.format_exc())
                 continue

            # Alignment Loss Calculation (CLIP-style)
            loss = alignment_loss_fn(visual_features_norm, target_semantics_norm)

            # Backward pass and optimization (only updates adapter)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'Align Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{config['training']['alignment']['epochs']}] Avg Align Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Alignment', avg_loss, epoch)
        if scheduler:
             writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        if scheduler:
            scheduler.step()

        # --- Save Checkpoint (Best loss and latest) ---
        if avg_loss < best_loss:
             best_loss = avg_loss
             best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')
             torch.save({
                 'epoch': epoch + 1,
                 'adapter_state_dict': hrrp_adapter.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_loss': best_loss,
                 'config': config
             }, best_ckpt_path)
             log.info(f"Best model checkpoint saved to {best_ckpt_path} (Loss: {best_loss:.4f})")

        latest_ckpt_path = os.path.join(ckpt_dir, 'latest.pth')
        torch.save({
            'epoch': epoch + 1,
            'adapter_state_dict': hrrp_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, latest_ckpt_path)
        # log.info(f"Latest model checkpoint saved to {latest_ckpt_path}") # Maybe too verbose per epoch

    writer.close()
    log.info("HRRP Adapter training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRRP 1D-to-2D Adapter")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file') # Updated default
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    main(configuration)
