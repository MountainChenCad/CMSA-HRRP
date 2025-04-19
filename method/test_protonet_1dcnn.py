# method/test_protonet_1dcnn.py
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
from data.samplers import CategoriesSampler
from model.hrrp_encoder import HRRPEncoder # Import the 1D CNN
from logger import loggers
from utils import set_seed, normalize, Cosine_classifier, proto_classifier, count_95acc, load_config, get_dynamic_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str, args_override: argparse.Namespace):
    """Evaluates ProtoNet baseline using a pre-trained 1D CNN."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config) # Paths might depend on baseline name config

    # --- Apply command-line overrides ---
    n_way = args_override.n_way if args_override.n_way is not None else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot is not None else config['fsl']['k_shot']
    q_query = config['fsl']['q_query']
    n_batch = config['fsl']['test_episodes']
    classifier_type = args_override.classifier if args_override.classifier else 'cosine' # Default to cosine
    classifier_temp = config['fsl'].get('classifier_temperature', 1.0) # Use 1.0 for standard cosine/euclidean

    # Determine checkpoint path for the baseline CNN encoder
    encoder_checkpoint_path = args_override.checkpoint if args_override.checkpoint else dynamic_paths['baseline_cnn_latest_ckpt'] # Or use best

    # --- Setup Logging ---
    fsl_setting_name = f"{n_way}way_{k_shot}shot_{classifier_type}"
    log_dir = os.path.join(dynamic_paths['protonet_test_log_dir'], fsl_setting_name)
    os.makedirs(log_dir, exist_ok=True)
    log = loggers(os.path.join(log_dir, f'test_log.txt'))

    log.info("Starting Few-Shot Learning Testing (ProtoNet Baseline)...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"FSL Setting: {n_way}-way {k_shot}-shot, {q_query}-query")
    log.info(f"Episodes: {n_batch}")
    log.info(f"Classifier: {classifier_type.upper()}")
    if classifier_type == 'cosine': log.info(f"Cosine Temp: {classifier_temp}")
    log.info(f"Using Encoder checkpoint: {encoder_checkpoint_path}")
    log.info(f"Logs will be saved to: {log_dir}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading (Novel Set) ---
    log.info("Loading novel dataset...")
    try:
        sim_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'simulated_hrrp')
        meas_path = os.path.join(config['paths'].get('datasets_base', './datasets'), 'measured_hrrp')
        novel_dataset = HRRPDataset(
            root_dirs=[sim_path, meas_path],
            split='novel',
            classes=config['data']['novel_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude'
        )
        if len(novel_dataset) == 0: log.error("Novel dataset is empty!"); sys.exit(1)

        # Check sample counts (copied from test_fsl.py)
        samples_per_class = {}
        for i in range(len(novel_dataset)):
             _, label_idx = novel_dataset[i]
             label_name = novel_dataset.idx_to_class[label_idx]
             samples_per_class[label_name] = samples_per_class.get(label_name, 0) + 1
        min_samples = k_shot + q_query
        valid_classes = [cls for cls, count in samples_per_class.items() if count >= min_samples]
        if len(valid_classes) < n_way:
            log.error(f"Not enough classes ({len(valid_classes)}) with sufficient samples ({min_samples}) for {n_way}-way {k_shot}-shot {q_query}-query evaluation.")
            log.error(f"Classes with counts: {samples_per_class}")
            sys.exit(1)
        log.info(f"{len(valid_classes)} novel classes have enough samples (>= {min_samples}).")


        novel_sampler = CategoriesSampler(
            novel_dataset.labels,
            n_batch=n_batch,
            n_cls=n_way,
            n_per=k_shot + q_query
        )
        log.info(f"Novel dataset loaded with {len(novel_dataset)} samples. Sampler initialized.")
    except Exception as e:
         log.error(f"Failed to initialize Novel HRRPDataset or Sampler: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained 1D CNN Encoder ---
    log.info(f"Loading pre-trained 1D CNN encoder from: {encoder_checkpoint_path}")
    try:
        if not os.path.exists(encoder_checkpoint_path):
             log.error(f"Encoder checkpoint not found: {encoder_checkpoint_path}. Run train_baseline_cnn.py first."); sys.exit(1)

        cnn_config = config.get('baseline_cnn', config['model'].get('hrrp_encoder', {}))
        encoder = HRRPEncoder(
            input_channels=1,
            output_dim=cnn_config.get('output_dim', 512),
            layers=cnn_config.get('layers', [64, 128, 256, 512]),
            kernel_size=cnn_config.get('kernel_size', 7)
        ).to(device)
        checkpoint_data = torch.load(encoder_checkpoint_path, map_location=device)
        # Load encoder state dict specifically
        encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
        encoder.eval() # Set to evaluation mode
        log.info(f"Baseline 1D CNN encoder loaded (from epoch {checkpoint_data.get('epoch', 'N/A')}).")
        feature_dim = cnn_config.get('output_dim', 512) # Get feature dimension
    except Exception as e:
        log.error(f"Error loading baseline CNN encoder: {e}", exc_info=True); sys.exit(1)

    # --- Testing Loop ---
    log.info(f"Starting testing on {n_batch} episodes...")
    all_episode_accuracies = []
    pbar = tqdm(range(n_batch), desc="Testing Episodes (ProtoNet)")
    episode_indices_generator = iter(novel_sampler)

    for episode_idx in pbar:
        try:
            # --- Sample Episode Data ---
            # Reusing sampling logic from test_fsl.py
            batch_indices = next(episode_indices_generator)
            support_indices = batch_indices[:n_way * k_shot]
            query_indices = batch_indices[n_way * k_shot:]

            support_data_list = []
            query_data_list = []
            query_labels_list = []
            current_episode_label = 0
            for i in range(n_way):
                cls_support_indices = support_indices[i*k_shot:(i+1)*k_shot]
                cls_query_indices = query_indices[i*q_query:(i+1)*q_query]
                for idx in cls_support_indices:
                     data, _ = novel_dataset[idx.item()]
                     support_data_list.append(data)
                for idx in cls_query_indices:
                     data, _ = novel_dataset[idx.item()]
                     query_data_list.append(data)
                     query_labels_list.append(current_episode_label) # Query labels 0 to N-1
                current_episode_label += 1

            support_data = torch.stack(support_data_list).to(device)
            query_data = torch.stack(query_data_list).to(device)
            query_labels_episode = torch.tensor(query_labels_list, device=device)

            # --- Feature Extraction (using baseline CNN) ---
            with torch.no_grad():
                support_features = encoder(support_data) # (N*K, D_feat)
                query_features = encoder(query_data)     # (N*Q, D_feat)

                # ProtoNet: Calculate prototypes by averaging support features per class
                support_features_reshaped = support_features.view(n_way, k_shot, feature_dim)
                prototypes = support_features_reshaped.mean(dim=1) # (N, D_feat)

            # --- Classification ---
            if classifier_type == 'cosine':
                # Normalize prototypes and query features for cosine similarity
                prototypes_norm = normalize(prototypes)
                query_features_norm = normalize(query_features)
                logits, predictions = Cosine_classifier(prototypes_norm, query_features_norm, temperature=classifier_temp)
            else: # Default to Euclidean
                # No normalization needed for Euclidean distance
                logits, predictions = proto_classifier(prototypes, query_features)

            # --- Calculate Accuracy ---
            acc = (predictions == query_labels_episode).float().mean().item()
            all_episode_accuracies.append(acc)
            pbar.set_postfix({'Acc': f"{acc:.4f}"})

        except StopIteration:
             log.warning("Sampler exhausted before reaching n_batch episodes.")
             break
        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {e}", exc_info=True)
            continue

    # --- Aggregate and Report Results ---
    if not all_episode_accuracies:
        log.error("No episodes were successfully completed. Cannot calculate accuracy.")
        sys.exit(1)

    accuracies_np = np.array(all_episode_accuracies)
    mean_acc, ci95 = count_95acc(accuracies_np)

    log.info(f"--- Testing Complete ({len(accuracies_np)} episodes) ---")
    log.info(f"{n_way}-way {k_shot}-shot ProtoNet Baseline Results:")
    log.info(f"  Classifier: {classifier_type.upper()}")
    log.info(f"  Mean Accuracy: {mean_acc * 100:.2f}%")
    log.info(f"  95% CI: +/- {ci95 * 100:.2f}%")

    # Save results summary
    results_path = os.path.join(log_dir, 'results_summary.yaml')
    try:
        with open(results_path, 'w') as f:
            yaml.dump({
                'config_path': config_path,
                'baseline': 'ProtoNet_1DCNN',
                'encoder_checkpoint': encoder_checkpoint_path,
                'fsl_setting': f"{n_way}w{k_shot}s",
                'classifier': classifier_type,
                'num_episodes': len(accuracies_np),
                'mean_accuracy': mean_acc,
                'ci95': ci95
            }, f, default_flow_style=False)
        log.info(f"Results summary saved to: {results_path}")
    except Exception as e:
        log.error(f"Failed to save results summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ProtoNet Baseline with 1D CNN")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    parser.add_argument('--n_way', type=int, default=None, help='Override N-way (default: from config)')
    parser.add_argument('--k_shot', type=int, default=None, help='Override K-shot (default: from config)')
    parser.add_argument('--classifier', type=str, choices=['cosine', 'euclidean'], default='cosine', help='Classifier type (default: cosine)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific encoder checkpoint (default: latest from dynamic path)')
    args = parser.parse_args()
    main(args.config, args)