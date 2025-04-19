# method/test_1dcnn_semantics.py
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
from model.hrrp_encoder import HRRPEncoder # Baseline encoder
from method.alignment import SemAlignModule # Fusion module
from logger import loggers
from utils import set_seed, normalize, Cosine_classifier, count_95acc, load_config, get_dynamic_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str, args_override: argparse.Namespace):
    """Evaluates 1D CNN + Semantics baseline."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Apply command-line overrides ---
    n_way = args_override.n_way if args_override.n_way is not None else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot is not None else config['fsl']['k_shot']
    q_query = config['fsl']['q_query']
    n_batch = config['fsl']['test_episodes']
    classifier_temp = config['fsl'].get('classifier_temperature', 10.0)
    # Kappa handling (similar to test_fsl.py)
    if args_override.kappa is not None:
        kappas_to_test = [args_override.kappa]
        kappa_mode = f"fixed_k{args_override.kappa}"
    else:
        kappas_to_test = config['fsl'].get('test_kappa_values')
        if kappas_to_test is None:
            kappas_to_test = np.linspace(0, 1, 11).tolist()
        else: kappas_to_test = [float(k) for k in kappas_to_test]
        kappa_mode = "sweep"

    # Determine checkpoint paths
    encoder_checkpoint_path = args_override.encoder_checkpoint if args_override.encoder_checkpoint else dynamic_paths['baseline_cnn_latest_ckpt']
    fusion_checkpoint_path = args_override.fusion_checkpoint if args_override.fusion_checkpoint else dynamic_paths['baseline_fusion_latest_ckpt']

    # --- Setup Logging ---
    fsl_setting_name = f"{n_way}way_{k_shot}shot_{kappa_mode}"
    log_dir = os.path.join(dynamic_paths['1dcnn_semantics_test_log_dir'], fsl_setting_name)
    os.makedirs(log_dir, exist_ok=True)
    log = loggers(os.path.join(log_dir, f'test_log.txt'))

    log.info("Starting Few-Shot Learning Testing (1D CNN + Semantics Baseline)...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"Text Type used for fusion: {config['semantics']['generation']['text_type']}")
    log.info(f"FSL Setting: {n_way}-way {k_shot}-shot, {q_query}-query")
    log.info(f"Episodes: {n_batch}")
    log.info(f"Classifier Temp: {classifier_temp}")
    log.info(f"Kappa values to test: {kappas_to_test}")
    log.info(f"Using Encoder checkpoint: {encoder_checkpoint_path}")
    log.info(f"Using Fusion checkpoint: {fusion_checkpoint_path}")
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

    # --- Load Semantic Features (z_T) ---
    semantic_feature_path = dynamic_paths['semantic_features']
    log.info(f"Loading semantic features (z_T) from: {semantic_feature_path}")
    try:
        if not os.path.exists(semantic_feature_path):
             log.error(f"Semantic features file not found: {semantic_feature_path}"); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features_dict = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        semantic_dim = config['model']['foundation_model']['text_encoder_dim']
        loaded_sem_dim = list(semantic_features_dict.values())[0].shape[-1]
        if loaded_sem_dim != semantic_dim: semantic_dim = loaded_sem_dim # Adjust if needed
        log.info(f"Loaded semantic features for {len(semantic_features_dict)} classes. Dim: {semantic_dim}")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained 1D CNN Encoder (Frozen) ---
    log.info(f"Loading pre-trained 1D CNN encoder from: {encoder_checkpoint_path}")
    try:
        if not os.path.exists(encoder_checkpoint_path):
             log.error(f"Encoder checkpoint not found: {encoder_checkpoint_path}"); sys.exit(1)
        cnn_config = config.get('baseline_cnn', config['model'].get('hrrp_encoder', {}))
        feature_dim = cnn_config.get('output_dim', 512) # Dim of z_H
        encoder = HRRPEncoder(input_channels=1, output_dim=feature_dim, layers=cnn_config.get('layers', [64, 128, 256, 512]), kernel_size=cnn_config.get('kernel_size', 7)).to(device)
        checkpoint_data = torch.load(encoder_checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
        encoder.eval()
        for param in encoder.parameters(): param.requires_grad = False
        log.info(f"Baseline 1D CNN encoder loaded and frozen. Feature Dim (z_H): {feature_dim}")
    except Exception as e:
        log.error(f"Error loading baseline CNN encoder: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained Fusion Module (Frozen, Optional) ---
    fusion_module = None
    if any(k > 0 for k in kappas_to_test):
        log.info(f"Loading pre-trained fusion module from: {fusion_checkpoint_path}")
        try:
            if not os.path.exists(fusion_checkpoint_path): log.warning(f"Baseline fusion checkpoint not found: {fusion_checkpoint_path}. Kappa>0 will be ignored."); raise FileNotFoundError
            fusion_module = SemAlignModule(
                visual_dim=feature_dim, # Input z_H dim
                semantic_dim=semantic_dim, # Input z_T dim
                hidden_dim=config['model']['fusion_module']['hidden_dim'],
                output_dim=feature_dim, # Output z_H dim
                drop=0.0
            ).to(device)
            fusion_checkpoint_data = torch.load(fusion_checkpoint_path, map_location=device)
            fusion_module.load_state_dict(fusion_checkpoint_data['fusion_1dcnn_state_dict'])
            fusion_module.eval()
            for param in fusion_module.parameters(): param.requires_grad = False
            log.info(f"Baseline fusion module loaded and frozen (from epoch {fusion_checkpoint_data.get('epoch', 'N/A')}).")
        except FileNotFoundError:
            fusion_module = None
            log.warning("Proceeding without baseline fusion module. Kappa values > 0 will be ineffective.")
            kappas_to_test = [0.0] if 0.0 not in kappas_to_test else kappas_to_test
        except Exception as e: log.error(f"Error loading baseline fusion module: {e}", exc_info=True); sys.exit(1)

    # --- Testing Loop ---
    log.info(f"Starting testing on {n_batch} episodes...")
    all_kappa_accuracies = {k: [] for k in kappas_to_test}
    pbar = tqdm(range(n_batch), desc="Testing Episodes (1DCNN+Sem)")
    episode_indices_generator = iter(novel_sampler)

    for episode_idx in pbar:
        try:
            # --- Sample Episode Data ---
            # Reusing sampling logic from test_fsl.py
            batch_indices = next(episode_indices_generator)
            support_indices = batch_indices[:n_way * k_shot]
            query_indices = batch_indices[n_way * k_shot:]
            support_data_list, query_data_list, query_labels_list = [], [], []
            episode_class_names = []
            label_map = {}
            current_episode_label = 0
            for i in range(n_way):
                cls_support_indices = support_indices[i*k_shot:(i+1)*k_shot]
                cls_query_indices = query_indices[i*q_query:(i+1)*q_query]
                first_sample_idx = cls_support_indices[0].item()
                _, original_label_idx = novel_dataset[first_sample_idx]
                class_name = novel_dataset.idx_to_class[original_label_idx]
                episode_class_names.append(class_name)
                label_map[original_label_idx] = current_episode_label
                for idx in cls_support_indices: support_data_list.append(novel_dataset[idx.item()][0])
                for idx in cls_query_indices:
                    query_data_list.append(novel_dataset[idx.item()][0])
                    query_labels_list.append(current_episode_label)
                current_episode_label += 1
            support_data = torch.stack(support_data_list).to(device)
            query_data = torch.stack(query_data_list).to(device)
            query_labels_episode = torch.tensor(query_labels_list, device=device)

            # --- Feature Extraction (z_H) ---
            with torch.no_grad():
                z_h_support = encoder(support_data) # (N*K, D_feat)
                z_h_query = encoder(query_data)     # (N*Q, D_feat)

                # Normalize features for prototype calculation and classification
                z_h_support_norm = normalize(z_h_support)
                z_h_query_norm = normalize(z_h_query)

                # Semantic features for the episode
                try:
                    z_t_episode = torch.stack([semantic_features_dict[name] for name in episode_class_names]).to(device)
                except KeyError as e:
                    log.warning(f"Episode {episode_idx}: Missing semantic feature for class {e}. Skipping."); continue

                # Calculate mean 1D CNN prototype (u_c) using normalized features
                z_h_support_norm_reshaped = z_h_support_norm.view(n_way, k_shot, feature_dim)
                u_c = z_h_support_norm_reshaped.mean(dim=1) # Shape: (N, D_feat)

                # Calculate reconstructed prototypes (r_c) if fusion module exists
                r_c = None
                if fusion_module is not None:
                    z_t_support_repeated = z_t_episode.repeat_interleave(k_shot, dim=0)
                    # Use UNNORMALIZED z_h_support as input to fusion module
                    reconstructed_features = fusion_module(z_h_support, z_t_support_repeated)
                    # Normalize the output before averaging
                    reconstructed_features_norm = normalize(reconstructed_features)
                    reconstructed_features_norm_reshaped = reconstructed_features_norm.view(n_way, k_shot, feature_dim)
                    r_c = reconstructed_features_norm_reshaped.mean(dim=1) # Shape: (N, D_feat)

            # --- Classification for each Kappa ---
            episode_accuracies = {}
            for current_kappa in kappas_to_test:
                with torch.no_grad():
                    # Determine prototype (p_c)
                    if current_kappa == 0 or r_c is None:
                        p_c = u_c # Use 1D CNN visual-only prototype
                    else:
                        # Fuse Prototypes: p_c = k * r_c + (1 - k) * u_c
                        p_c = current_kappa * r_c + (1 - current_kappa) * u_c
                        p_c = normalize(p_c) # Normalize fused prototype

                    # Classification using normalized query features (z_h_query_norm)
                    logits, predictions = Cosine_classifier(p_c, z_h_query_norm, temperature=classifier_temp)

                    # Calculate Accuracy
                    acc = (predictions == query_labels_episode).float().mean().item()
                    episode_accuracies[current_kappa] = acc
                    all_kappa_accuracies[current_kappa].append(acc)

            pbar.set_postfix({f'Acc(k={k:.1f})': f"{acc:.4f}" for k, acc in episode_accuracies.items()})

        except StopIteration:
             log.warning("Sampler exhausted before reaching n_batch episodes."); break
        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {e}", exc_info=True); continue

    # --- Aggregate and Report Results ---
    log.info(f"--- Testing Complete ({len(list(all_kappa_accuracies.values())[0])} episodes processed) ---")
    log.info(f"{n_way}-way {k_shot}-shot (1D CNN + Semantics) Baseline Results:")

    best_mean_acc, best_kappa, best_ci95 = -1.0, -1.0, 0.0
    results_summary = {}
    for k in kappas_to_test:
        accuracies_np = np.array(all_kappa_accuracies[k])
        if len(accuracies_np) == 0: continue
        mean_acc, ci95 = count_95acc(accuracies_np)
        log.info(f"  Kappa = {k:.2f}: Mean Accuracy = {mean_acc * 100:.2f}% +/- {ci95 * 100:.2f}%")
        results_summary[k] = {'mean': mean_acc, 'ci95': ci95}
        if mean_acc > best_mean_acc: best_mean_acc, best_kappa, best_ci95 = mean_acc, k, ci95

    log.info(f"--- Best Mean Accuracy across tested kappas ---")
    log.info(f"  Best Kappa: {best_kappa:.2f}")
    log.info(f"  Mean Accuracy: {best_mean_acc * 100:.2f}%")
    log.info(f"  95% CI: +/- {best_ci95 * 100:.2f}%")

    # Save results summary
    results_path = os.path.join(log_dir, 'results_summary.yaml')
    try:
        with open(results_path, 'w') as f:
            yaml.dump({
                'config_path': config_path,
                'baseline': '1DCNN_Semantics',
                'text_type': config['semantics']['generation']['text_type'],
                'encoder_checkpoint': encoder_checkpoint_path,
                'fusion_checkpoint': fusion_checkpoint_path if fusion_module else 'N/A',
                'fsl_setting': f"{n_way}w{k_shot}s",
                'results_per_kappa': results_summary,
                'best_result': {'kappa': best_kappa, 'mean_acc': best_mean_acc, 'ci95': best_ci95}
            }, f, default_flow_style=False)
        log.info(f"Results summary saved to: {results_path}")
    except Exception as e: log.error(f"Failed to save results summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1D CNN + Semantics Baseline")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    parser.add_argument('--n_way', type=int, default=None, help='Override N-way')
    parser.add_argument('--k_shot', type=int, default=None, help='Override K-shot')
    parser.add_argument('--kappa', type=float, default=None, help='Test a specific kappa value')
    parser.add_argument('--encoder_checkpoint', type=str, default=None, help='Path to specific 1D CNN encoder checkpoint')
    parser.add_argument('--fusion_checkpoint', type=str, default=None, help='Path to specific baseline fusion module checkpoint')
    args = parser.parse_args()
    main(args.config, args)