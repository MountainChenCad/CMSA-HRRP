# method/train_baseline_cnn.py
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
from model.hrrp_encoder import HRRPEncoder # Import the 1D CNN
from logger import loggers
from utils import set_seed, load_config, get_dynamic_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str):
    """Trains a baseline 1D CNN (HRRPEncoder) for classification."""
    config = load_config(config_path)
    # Note: Baseline paths might be less dynamic if baseline is fixed, adjust get_dynamic_paths if needed
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup ---
    log_dir = dynamic_paths['baseline_cnn_log_dir']
    ckpt_dir = dynamic_paths['baseline_cnn_checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'train_baseline_cnn_log.txt'))
    log.info("Starting Baseline 1D CNN training...")
    log.info(f"Loaded configuration from: {config_path}")
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
            phase_info='magnitude' # Assuming 1D CNN takes magnitude
        )
        num_base_classes = len(config['data']['base_classes'])
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
    log.info(f"Base dataset loaded with {len(base_dataset)} samples and {num_base_classes} classes.")

    # --- Model Initialization ---
    # Use a separate config section for baseline CNN if hyperparameters differ significantly
    cnn_config = config.get('baseline_cnn', config['model'].get('hrrp_encoder', {})) # Fallback to old config structure
    output_dim = cnn_config.get('output_dim', 512)
    # Add a classification head
    model = HRRPEncoder(
        input_channels=1, # Assuming magnitude
        output_dim=output_dim, # Feature dimension
        layers=cnn_config.get('layers', [64, 128, 256, 512]),
        kernel_size=cnn_config.get('kernel_size', 7)
        # Add other HRRPEncoder params if needed
    )
    classifier_head = nn.Linear(output_dim, num_base_classes) # Simple linear classifier
    full_model = nn.Sequential(model, classifier_head).to(device)
    log.info(f"Initialized Baseline 1D CNN model: {model}")
    log.info(f"Added Classifier Head (Output: {num_base_classes})")

    # --- Optimizer and Loss ---
    params_to_optimize = list(full_model.parameters())
    opt_name = config['training']['optimizer'].lower()
    if opt_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    elif opt_name == 'adamw':
         optimizer = optim.AdamW(params_to_optimize, lr=config['training']['lr'])
    else:
        log.warning(f"Unsupported optimizer: {opt_name}. Using Adam.")
        optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])

    criterion = nn.CrossEntropyLoss()
    log.info("Using CrossEntropyLoss for classification.")

    # --- LR Scheduler (Optional) ---
    scheduler = None # Add scheduler setup if needed

    # --- Training Loop ---
    log.info("Starting baseline CNN training loop...")
    best_loss = float('inf') # Or use validation accuracy if implementing validation split
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        full_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(enumerate(base_loader), total=len(base_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for step, (hrrp_samples, labels) in pbar:
            hrrp_samples = hrrp_samples.to(device)
            labels = labels.to(device) # Labels should be 0 to num_base_classes-1

            optimizer.zero_grad()
            try:
                logits = full_model(hrrp_samples)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix({'CE Loss': f"{loss.item():.4f}"})

            except Exception as e:
                log.error(f"Error during training step {step}: {e}", exc_info=True)
                optimizer.zero_grad()
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        log.info(f"[Epoch {epoch+1}/{epochs}] Avg CE Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        writer.add_scalar('Loss/CrossEntropy', avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', accuracy, epoch)
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('LR', current_lr, epoch)
            log.info(f"Current LR: {current_lr:.6f}")
            scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', current_lr, epoch)

        # --- Save Checkpoint (Save only the HRRPEncoder part for FSL use) ---
        # Use loss as the metric for saving best model here
        is_best = avg_loss < best_loss
        if is_best:
             best_loss = avg_loss
             log.info(f"New best loss: {best_loss:.4f}")

        # Save latest checkpoint (encoder only)
        latest_ckpt_path = dynamic_paths['baseline_cnn_latest_ckpt']
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': model.state_dict(), # Save only encoder state
            'loss': avg_loss,
            'accuracy': accuracy,
            'config': config # Save config for reference
        }, latest_ckpt_path)

        # Save best checkpoint (encoder only)
        if is_best:
            best_ckpt_path = dynamic_paths['baseline_cnn_best_ckpt']
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': model.state_dict(), # Save only encoder state
                'best_loss': best_loss,
                 'accuracy': accuracy,
                'config': config
            }, best_ckpt_path)
            log.info(f"Best baseline encoder model saved to {best_ckpt_path}")

    writer.close()
    log.info("Baseline 1D CNN training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 1D CNN (HRRPEncoder)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()
    main(args.config)