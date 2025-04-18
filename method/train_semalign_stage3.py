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
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from method.alignment import SemAlignModule # Use the renamed module
from logger import loggers
from utils import set_seed, normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(config):
    """Main training function for SemAlign Module (Stage 3)."""

    # --- Setup ---
    log_dir = os.path.join(config['paths'].get('logs', './logs'), 'semalign_training_stage3')
    ckpt_dir = os.path.join(config['paths'].get('checkpoints', './checkpoints'), 'semalign_module') # New checkpoint dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_semalign'))
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
            phase_info='magnitude' # Match adapter input expectation
        )
    except Exception as e:
         log.error(f"Failed to initialize Base HRRPDataset: {e}")
         sys.exit(1)
    base_loader = DataLoader(
        base_dataset,
        batch_size=config['training']['alignment']['batch_size'], # Reuse batch size
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

    # --- Load Pre-computed Base Centers ---
    center_file_path = config.get('paths', {}).get('base_centers_path', './checkpoints/base_centers_mean.pth')
    log.info(f"Loading pre-computed base centers from: {center_file_path}")
    try:
        center_data = torch.load(center_file_path, map_location='cpu')
        base_centers = {k: v.float().to(device) for k, v in center_data['center_mean'].items()}
        log.info(f"Loaded base centers for {len(base_centers)} classes.")
        # Infer visual dimension from centers
        visual_dim = list(base_centers.values())[0].shape[0]
        log.info(f"Inferred visual dimension from centers: {visual_dim}")
    except FileNotFoundError:
        log.error(f"Base centers file not found at {center_file_path}. Run compute_centers.py first.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error loading base centers: {e}")
        sys.exit(1)

    # --- Load VLM Visual Encoder (Frozen) ---
    fm_config = config['model']['foundation_model']
    log.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            fm_variant = fm_config['variant']
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"
            local_weights_path = os.path.join(base_checkpoint_dir, 'foundation_models', weights_filename)
            if not os.path.exists(local_weights_path): log.error(f"VLM weights not found: {local_weights_path}"); sys.exit(1)
            log.info(f"Loading VLM weights from: {local_weights_path}")

            vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_variant, pretrained=local_weights_path)
            visual_encoder = vlm_model.visual.to(device).eval() # f_V
            for param in visual_encoder.parameters(): param.requires_grad = False
            log.info("VLM Visual Encoder loaded and frozen.")
            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
            if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
        else: log.error(f"Unsupported VLM: {fm_config['name']}"); sys.exit(1)
    except Exception as e: log.error(f"Failed to load VLM: {e}"); sys.exit(1)

    # --- Load Pre-trained Adapter (Frozen) ---
    adapter_checkpoint_path = os.path.join(config['paths']['checkpoints'], 'hrrp_adapter', 'latest.pth')
    log.info(f"Loading pre-trained adapter from: {adapter_checkpoint_path}")
    try:
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'], input_channels=1, output_channels=3, output_size=expected_img_size
        ).to(device)
        adapter_checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint['adapter_state_dict'])
        adapter.eval()
        for param in adapter.parameters(): param.requires_grad = False # Freeze adapter
        log.info("Pre-trained adapter loaded and frozen.")
    except Exception as e: log.error(f"Error loading adapter: {e}"); sys.exit(1)

    # --- Initialize SemAlign Module (Trainable) ---
    semantic_dim = config['model']['foundation_model']['text_encoder_dim']
    # Ensure visual_dim matches the dimension of the loaded centers
    if visual_dim != config['model']['foundation_model'].get('visual_encoder_dim', semantic_dim):
         log.warning(f"Visual dim from centers ({visual_dim}) differs from config ({config['model']['foundation_model'].get('visual_encoder_dim', semantic_dim)}). Using center dim.")
         # Update config value in memory if needed, though visual_dim variable is used below

    semalign_module = SemAlignModule(
        visual_dim=visual_dim, # Use inferred dim from centers
        semantic_dim=semantic_dim,
        hidden_dim=config['model']['fusion_module']['hidden_dim'], # Reuse fusion hidden dim
        output_dim=visual_dim, # Output matches visual center dim
        drop=config['training']['alignment'].get('dropout_semalign', 0.2) # Allow separate dropout config
    ).to(device)

    # --- Optimizer ---
    # Optimize ONLY the SemAlign module parameters
    params_to_optimize = list(semalign_module.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config['training']['alignment']['lr']) # Reuse LR setting

    # Loss function: L1 distance as per SemFew Eq. 4
    recon_loss_fn = nn.L1Loss()

    # LR scheduler (optional)
    scheduler = None # ... (scheduler setup if needed) ...

    # --- Training Loop ---
    log.info("Starting SemAlign Module training (Stage 3)...")
    best_loss = float('inf')
    for epoch in range(config['training']['alignment']['epochs']): # Reuse epoch count
        semalign_module.train() # Set SemAlign to train mode

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{config['training']['alignment']['epochs']}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device)
            labels = labels.to(device)

            # Get corresponding semantic features and target centers
            try:
                target_semantics = torch.stack([semantic_features[base_dataset.idx_to_class[l.item()]] for l in labels])
                target_centers = torch.stack([base_centers[base_dataset.idx_to_class[l.item()]] for l in labels])
            except KeyError as e:
                log.warning(f"KeyError accessing semantic/center for label {e}. Skipping batch {step}.")
                continue

            # Forward pass through frozen adapter and visual encoder
            with torch.no_grad():
                pseudo_images = adapter(hrrp_samples)
                if pseudo_images.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images = F.interpolate(pseudo_images, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                visual_features = visual_encoder(pseudo_images) # z_V
                if isinstance(visual_features, tuple): visual_features = visual_features[0]
                if visual_features.dim() == 3 and visual_features.shape[1] > 1: visual_features = visual_features[:, 0]
                # NO normalization here for z_V input to SemAlign, matching SemFew's f(xi) input

            # Forward pass through SemAlign module (trainable)
            optimizer.zero_grad()
            reconstructed_centers = semalign_module(visual_features, target_semantics) # r = h([z_V, z_T])

            # Reconstruction Loss Calculation (SemFew Eq. 4)
            loss = recon_loss_fn(reconstructed_centers, target_centers) # L1 distance to mean center

            # Backward pass and optimization (only updates SemAlign module)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'Recon Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        log.info(f"[Epoch {epoch+1}/{config['training']['alignment']['epochs']}] Avg Recon Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/Reconstruction', avg_loss, epoch)
        if scheduler: writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        if scheduler: scheduler.step()

        # --- Save Checkpoint ---
        if avg_loss < best_loss:
             best_loss = avg_loss
             best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')
             torch.save({
                 'epoch': epoch + 1,
                 'semalign_state_dict': semalign_module.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_loss': best_loss,
                 'config': config # Save config for reference
             }, best_ckpt_path)
             log.info(f"Best SemAlign model saved to {best_ckpt_path} (Loss: {best_loss:.4f})")

        latest_ckpt_path = os.path.join(ckpt_dir, 'latest.pth')
        torch.save({
            'epoch': epoch + 1,
            'semalign_state_dict': semalign_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, latest_ckpt_path)

    writer.close()
    log.info("SemAlign Module training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SemAlign Module (Stage 3)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config): logger.error(f"Config not found: {args.config}"); sys.exit(1)
    configuration = load_config(args.config)
    # Ensure paths for centers and adapter are defined
    if 'paths' not in configuration: configuration['paths'] = {}
    if 'base_centers_path' not in configuration['paths']: configuration['paths']['base_centers_path'] = './checkpoints/base_centers_mean.pth'
    if 'checkpoints' not in configuration['paths']: configuration['paths']['checkpoints'] = './checkpoints'

    main(configuration)