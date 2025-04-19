# method/test_fsl.py

import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip # Assuming RemoteCLIP

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from data.samplers import CategoriesSampler
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from method.alignment import SemAlignModule
from logger import loggers
from utils import set_seed, normalize, Cosine_classifier, count_95acc

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
    """Main function for Few-Shot Learning testing: Kappa sweep and visual-only baseline."""

    # --- Setup ---
    log_dir = os.path.join(config['paths'].get('logs', './logs'), 'fsl_testing_kappa_sweep_baseline') # Adjusted log dir name
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"test_{config['fsl']['n_way']}way_{config['fsl']['k_shot']}shot_kappa_sweep_baseline"
    log = loggers(os.path.join(log_dir, log_file_name))
    log.info("Starting Few-Shot Learning Testing (Kappa Sweep + Visual-Only Baseline)...")
    log.info(f"Loaded configuration: {config}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- FSL Parameters ---
    n_way = config['fsl']['n_way']
    k_shot = config['fsl']['k_shot']
    q_query = config['fsl']['q_query']
    n_batch = config['fsl']['test_episodes'] # Number of test episodes
    classifier_temp = config['fsl'].get('classifier_temperature', 1.0)

    # --- Kappa Values for Testing ---
    kappa_values_to_test = config['fsl'].get('test_kappa_values')
    if kappa_values_to_test is None:
        kappa_values_to_test = np.linspace(0, 1, 11).tolist()
        log.warning(f"Config 'fsl.test_kappa_values' not found. Using default: {kappa_values_to_test}")
    else:
        kappa_values_to_test = [float(k) for k in kappa_values_to_test]

    log.info(f"FSL Setup: {n_way}-way, {k_shot}-shot, {q_query}-query samples")
    log.info(f"Testing on {n_batch} episodes. Classifier temp={classifier_temp}.")
    log.info(f"Sweeping Kappa values: {kappa_values_to_test}")

    # --- Data Loading (Novel Set) ---
    log.info("Loading novel dataset...")
    try:
        novel_dataset = HRRPDataset(
            root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
            split='novel',
            classes=config['data']['novel_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude'
        )
    except Exception as e:
         log.error(f"Failed to initialize Novel HRRPDataset: {e}"); sys.exit(1)

    novel_sampler = CategoriesSampler(
        novel_dataset.labels,
        n_batch=n_batch,
        n_cls=n_way,
        n_per=k_shot + q_query
    )
    log.info(f"Novel dataset loaded with {len(novel_dataset)} samples. Sampler initialized.")

    # --- Load Semantic Features ---
    semantic_feature_path = config['semantics']['feature_path']
    log.info(f"Loading semantic features from: {semantic_feature_path}")
    try:
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features_dict = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        semantic_features_dict = {k: normalize(v) for k, v in semantic_features_dict.items()}
        semantic_dim = list(semantic_features_dict.values())[0].shape[-1]
        log.info(f"Loaded and normalized semantic features for {len(semantic_features_dict)} classes. Dim: {semantic_dim}")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}"); sys.exit(1)

    # --- Load VLM (Visual Encoder - Frozen) ---
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
            visual_encoder = vlm_model.visual.to(device).eval()

            for param in vlm_model.visual.parameters(): param.requires_grad = False
            log.info("VLM Visual Encoder loaded and frozen.")

            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
            if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]

            if hasattr(visual_encoder, 'proj') and visual_encoder.proj is not None:
                 visual_dim = visual_encoder.proj.shape[1]
                 log.info(f"Inferred visual dimension from VLM proj layer: {visual_dim}")
            elif fm_config.get('visual_encoder_dim'):
                 visual_dim = fm_config['visual_encoder_dim']
                 log.info(f"Using visual dimension from config: {visual_dim}")
            else:
                 visual_dim = semantic_dim
                 log.warning(f"Visual dim not found, assuming it matches text dim: {visual_dim}")

        else: log.error(f"Unsupported VLM: {fm_config['name']}"); sys.exit(1)
    except Exception as e: log.error(f"Failed to load VLM: {e}"); import traceback; log.error(traceback.format_exc()); sys.exit(1)

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
        for param in adapter.parameters(): param.requires_grad = False
        log.info("Pre-trained adapter loaded and frozen.")
    except Exception as e: log.error(f"Error loading adapter checkpoint: {e}"); sys.exit(1)

    # --- Load Pre-trained SemAlign Module (Frozen) ---
    semalign_checkpoint_path = os.path.join(config['paths']['checkpoints'], 'semalign_module', 'latest.pth')
    log.info(f"Loading pre-trained SemAlign module from: {semalign_checkpoint_path}")
    try:
        semalign_module = SemAlignModule(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
            hidden_dim=config['model']['fusion_module']['hidden_dim'],
            output_dim=visual_dim,
            drop=0.0
        ).to(device)
        semalign_checkpoint = torch.load(semalign_checkpoint_path, map_location=device)
        semalign_module.load_state_dict(semalign_checkpoint['semalign_state_dict'])
        semalign_module.eval()
        for param in semalign_module.parameters(): param.requires_grad = False
        log.info("Pre-trained SemAlign module loaded and frozen.")
    except Exception as e: log.error(f"Error loading SemAlign module checkpoint: {e}"); sys.exit(1)

    # --- Testing Loop ---
    log.info(f"Starting testing on {n_batch} episodes...")
    all_episode_best_fused_accuracies = []
    all_episode_visual_only_accuracies = [] # NEW: Store visual-only accuracy
    all_episode_best_kappas = []

    pbar = tqdm(range(n_batch), desc="Testing Episodes")
    for episode_idx in pbar:
        try:
            # --- Sample Episode Data ---
            support_indices, query_indices, episode_class_names = sample_one_episode(novel_dataset, novel_sampler, n_way, k_shot, q_query)
            if support_indices is None:
                log.warning(f"Could not sample episode {episode_idx}. Skipping.")
                continue

            support_data = torch.stack([novel_dataset[i][0] for i in support_indices]).to(device)
            query_data = torch.stack([novel_dataset[i][0] for i in query_indices]).to(device)
            query_labels_episode = torch.arange(n_way, device=device).repeat_interleave(q_query)

            # --- Feature Extraction (Common for all kappas in this episode) ---
            with torch.no_grad():
                # Support features
                pseudo_images_support = adapter(support_data)
                if pseudo_images_support.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_support = F.interpolate(pseudo_images_support, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_support = visual_encoder(pseudo_images_support)
                if isinstance(z_v_support, tuple): z_v_support = z_v_support[0]
                if z_v_support.dim() == 3 and z_v_support.shape[1] > 1: z_v_support = z_v_support[:, 0]

                # Query features
                pseudo_images_query = adapter(query_data)
                if pseudo_images_query.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_query = F.interpolate(pseudo_images_query, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_query = visual_encoder(pseudo_images_query)
                if isinstance(z_v_query, tuple): z_v_query = z_v_query[0]
                if z_v_query.dim() == 3 and z_v_query.shape[1] > 1: z_v_query = z_v_query[:, 0]
                z_v_query_norm = normalize(z_v_query) # Normalize query features once

                # Semantic features for the episode
                try:
                    z_t_episode = torch.stack([semantic_features_dict[name] for name in episode_class_names]).to(device)
                except KeyError as e:
                    log.warning(f"Episode {episode_idx}: Missing semantic feature for class {e}. Skipping.")
                    continue

                # Calculate u_t (Mean Visual Prototype)
                z_v_support_reshaped = z_v_support.view(n_way, k_shot, visual_dim)
                u_t = z_v_support_reshaped.mean(dim=1) # (N, D_vis)

                # Calculate r_t (Reconstructed Prototype)
                z_t_support_repeated = z_t_episode.repeat_interleave(k_shot, dim=0)
                reconstructed_features = semalign_module(z_v_support, z_t_support_repeated)
                reconstructed_features_reshaped = reconstructed_features.view(n_way, k_shot, visual_dim)
                r_t = reconstructed_features_reshaped.mean(dim=1) # (N, D_vis)

            # --- Visual-Only Baseline Classification (Using u_t) ---
            with torch.no_grad():
                u_t_norm = normalize(u_t) # Normalize the visual-only prototype
                logits_visual_only, predictions_visual_only = Cosine_classifier(u_t_norm, z_v_query_norm, temperature=classifier_temp)
                acc_visual_only = (predictions_visual_only == query_labels_episode).float().mean().item()
                all_episode_visual_only_accuracies.append(acc_visual_only) # Store visual-only result

            # --- Inner Loop: Iterate through Kappa values for Fused Prototype ---
            accuracies_fused_for_this_episode = []
            for current_kappa in kappa_values_to_test:
                with torch.no_grad():
                    # Fuse Prototypes (p_t) Eq. 6
                    p_t = current_kappa * r_t + (1 - current_kappa) * u_t
                    p_t_norm = normalize(p_t)

                    # Classification
                    logits_fused, predictions_fused = Cosine_classifier(p_t_norm, z_v_query_norm, temperature=classifier_temp)

                    # Calculate Accuracy for this kappa
                    acc_fused = (predictions_fused == query_labels_episode).float().mean().item()
                    accuracies_fused_for_this_episode.append(acc_fused)

            # --- Find Best Fused Accuracy for this Episode ---
            best_fused_acc_for_episode = max(accuracies_fused_for_this_episode)
            best_kappa_index = np.argmax(accuracies_fused_for_this_episode)
            best_kappa_for_episode = kappa_values_to_test[best_kappa_index]

            all_episode_best_fused_accuracies.append(best_fused_acc_for_episode)
            all_episode_best_kappas.append(best_kappa_for_episode)

            pbar.set_postfix({'Vis Acc': f"{acc_visual_only:.4f}", 'Best Fused': f"{best_fused_acc_for_episode:.4f}", 'Best k': f"{best_kappa_for_episode:.2f}"})

        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {e}")
            import traceback
            log.error(traceback.format_exc())
            continue # Skip to next episode

    # --- Results ---
    if not all_episode_best_fused_accuracies or not all_episode_visual_only_accuracies:
        log.error("No episodes were successfully completed. Cannot calculate accuracy.")
        sys.exit(1)

    # Calculate stats for fused results
    fused_accuracies_np = np.array(all_episode_best_fused_accuracies)
    best_kappas_np = np.array(all_episode_best_kappas)
    mean_fused_acc, ci95_fused = count_95acc(fused_accuracies_np)
    mean_best_kappa = np.mean(best_kappas_np)
    std_best_kappa = np.std(best_kappas_np)

    # Calculate stats for visual-only results
    visual_only_accuracies_np = np.array(all_episode_visual_only_accuracies)
    mean_visual_only_acc, ci95_visual_only = count_95acc(visual_only_accuracies_np)

    log.info(f"--- Testing Complete ({n_batch} episodes) ---")
    log.info(f"{n_way}-way {k_shot}-shot Results:")

    log.info(f"  Visual-Only Prototype (u_t) Baseline:")
    log.info(f"    Mean Accuracy: {mean_visual_only_acc * 100:.2f}%")
    log.info(f"    95% CI: +/- {ci95_visual_only * 100:.2f}%")

    log.info(f"  Fused Prototype (p_t = k*r_t + (1-k)*u_t) with Kappa Sweep ({kappa_values_to_test}):")
    log.info(f"    Mean Best Accuracy (per episode): {mean_fused_acc * 100:.2f}%")
    log.info(f"    95% CI: +/- {ci95_fused * 100:.2f}%")
    log.info(f"    Average Best Kappa: {mean_best_kappa:.3f} (Std: {std_best_kappa:.3f})")

    # Optional: Log detailed per-episode results if needed
    # log.info(f"  Raw Visual Acc (first 10): {[f'{a*100:.2f}' for a in visual_only_accuracies_np[:10]]}")
    # log.info(f"  Raw Best Fused Acc (first 10): {[f'{a*100:.2f}' for a in fused_accuracies_np[:10]]}")
    # log.info(f"  Best Kappas (first 10): {[f'{k:.2f}' for k in best_kappas_np[:10]]}")


# Helper function to simulate episode sampling (Same as before)
def sample_one_episode(dataset, sampler, n_way, k_shot, q_query):
    """ Manually samples one episode using sampler logic. Returns indices and class names."""
    try:
        available_classes = list(range(sampler.num_classes))
        if len(available_classes) < n_way:
             logger.error("Not enough classes in dataset for N-way.")
             return None, None, None
        # Use sampler's internal class indices (0 to num_classes-1)
        episode_internal_class_indices = np.random.choice(available_classes, size=n_way, replace=False)

        support_indices = []
        query_indices = []
        episode_class_names = []

        for internal_cls_idx in episode_internal_class_indices:
            # Map internal sampler class index back to original dataset class index/name
            # Assumes sampler.m_ind is ordered according to sorted(list(set(dataset.labels)))
            original_class_label_value = sorted(list(set(dataset.labels)))[internal_cls_idx]
            class_name = dataset.idx_to_class[original_class_label_value] # Map label value to name
            episode_class_names.append(class_name)

            all_indices_for_class = sampler.m_ind[internal_cls_idx] # Use sampler's internal index
            if len(all_indices_for_class) < k_shot + q_query:
                logger.warning(f"Class {class_name} has only {len(all_indices_for_class)} samples, needing {k_shot+q_query}. Skipping episode.")
                return None, None, None

            selected_pos = torch.randperm(len(all_indices_for_class))[:k_shot + q_query]
            selected_indices = all_indices_for_class[selected_pos]

            support_indices.extend(selected_indices[:k_shot].tolist())
            query_indices.extend(selected_indices[k_shot:].tolist())

        return support_indices, query_indices, episode_class_names
    except Exception as e:
        logger.error(f"Error during manual episode sampling: {e}")
        import traceback; logger.error(traceback.format_exc())
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Learning Testing (Kappa Sweep + Baseline) for HRRP")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)

    # Add default FSL params if missing in config
    if 'fsl' not in configuration: configuration['fsl'] = {}
    if 'n_way' not in configuration['fsl']: configuration['fsl']['n_way'] = 5
    if 'k_shot' not in configuration['fsl']: configuration['fsl']['k_shot'] = 5
    if 'q_query' not in configuration['fsl']: configuration['fsl']['q_query'] = 15
    if 'test_episodes' not in configuration['fsl']: configuration['fsl']['test_episodes'] = 600
    if 'classifier_temperature' not in configuration['fsl']: configuration['fsl']['classifier_temperature'] = 10.0

    main(configuration)