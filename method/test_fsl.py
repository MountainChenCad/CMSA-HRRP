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
import open_clip

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from data.samplers import CategoriesSampler
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from method.alignment import SemAlignModule
from logger import loggers
from utils import set_seed, normalize, Cosine_classifier, count_95acc, load_config, get_dynamic_paths # Import helpers

logging.basicConfig(level=logging.INFO) # Basic config for early messages
logger = logging.getLogger(__name__)

# Helper function to sample episode (copied and potentially adapted)
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
        episode_class_names = [] # Store original class names from config

        for internal_cls_idx in episode_internal_class_indices:
            # Map internal sampler class index back to original dataset class index/name
            # Assumes sampler.m_ind is ordered according to sorted(list(set(dataset.labels)))
            original_class_label_value = sorted(list(set(dataset.labels)))[internal_cls_idx]
            class_name = dataset.idx_to_class[original_class_label_value] # Map label value to name
            episode_class_names.append(class_name)

            all_indices_for_class = sampler.m_ind[internal_cls_idx] # Use sampler's internal index
            if len(all_indices_for_class) < k_shot + q_query:
                logger.warning(f"Class {class_name} has only {len(all_indices_for_class)} samples, needing {k_shot+q_query}. Trying to sample with replacement for query or skipping.")
                # Option 1: Sample with replacement for query (might reuse support samples)
                if len(all_indices_for_class) < k_shot:
                     logger.error(f"Cannot even get {k_shot} support samples for {class_name}. Skipping episode.")
                     return None, None, None
                support_pos = torch.randperm(len(all_indices_for_class))[:k_shot]
                query_pos = torch.randint(0, len(all_indices_for_class), (q_query,)) # Sample query indices with replacement
                selected_support_indices = all_indices_for_class[support_pos]
                selected_query_indices = all_indices_for_class[query_pos]
                support_indices.extend(selected_support_indices.tolist())
                query_indices.extend(selected_query_indices.tolist())
                # Option 2: Skip episode (safer)
                # logger.warning(f"Skipping episode due to insufficient samples for {class_name}.")
                # return None, None, None
            else:
                # Standard sampling without replacement
                selected_pos = torch.randperm(len(all_indices_for_class))[:k_shot + q_query]
                selected_indices = all_indices_for_class[selected_pos]
                support_indices.extend(selected_indices[:k_shot].tolist())
                query_indices.extend(selected_indices[k_shot:].tolist())

        return support_indices, query_indices, episode_class_names
    except Exception as e:
        logger.error(f"Error during manual episode sampling: {e}", exc_info=True)
        return None, None, None


def main(config_path: str, args_override: argparse.Namespace):
    """Main function for Few-Shot Learning testing using Adapter approach."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Apply command-line overrides ---
    n_way = args_override.n_way if args_override.n_way is not None else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot is not None else config['fsl']['k_shot']
    q_query = config['fsl']['q_query'] # Usually not overridden
    n_batch = config['fsl']['test_episodes'] # Usually not overridden
    classifier_temp = config['fsl'].get('classifier_temperature', 10.0)
    # Kappa handling: single value from args, default from config, or sweep
    if args_override.kappa is not None:
        kappas_to_test = [args_override.kappa] # Test only specified kappa
        kappa_mode = f"fixed_k{args_override.kappa}"
    else:
        # Default: Use kappa sweep defined in config or default linspace
        kappas_to_test = config['fsl'].get('test_kappa_values')
        if kappas_to_test is None:
            kappas_to_test = np.linspace(0, 1, 11).tolist() # Default sweep
            logger.warning(f"Config 'fsl.test_kappa_values' not found. Using default sweep: {kappas_to_test}")
        else:
            kappas_to_test = [float(k) for k in kappas_to_test]
        kappa_mode = "sweep"

    # Determine checkpoint path for adapter and semalign
    adapter_checkpoint_path = args_override.checkpoint if args_override.checkpoint else dynamic_paths['adapter_latest_ckpt']
    # Semalign checkpoint depends on the adapter used for training it
    semalign_checkpoint_path = dynamic_paths['semalign_latest_ckpt'] # Assuming latest SemAlign corresponds to latest adapter

    # --- Setup Logging ---
    fsl_setting_name = f"{n_way}way_{k_shot}shot_{kappa_mode}"
    log_dir = os.path.join(dynamic_paths['fsl_test_log_dir'], fsl_setting_name)
    os.makedirs(log_dir, exist_ok=True)
    log = loggers(os.path.join(log_dir, f'test_log.txt'))

    log.info("Starting Few-Shot Learning Testing (Adapter Approach)...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"VLM Variant: {config['model']['foundation_model']['variant']}")
    log.info(f"Text Type: {config['semantics']['generation']['text_type']}")
    log.info(f"FSL Setting: {n_way}-way {k_shot}-shot, {q_query}-query")
    log.info(f"Episodes: {n_batch}")
    log.info(f"Classifier Temp: {classifier_temp}")
    log.info(f"Kappa values to test: {kappas_to_test}")
    log.info(f"Using Adapter checkpoint: {adapter_checkpoint_path}")
    log.info(f"Using SemAlign checkpoint: {semalign_checkpoint_path}")
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
        if len(novel_dataset) == 0:
             log.error("Novel dataset is empty! Check paths and class names.")
             sys.exit(1)

        # Ensure enough samples per class for the FSL setting
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
        # Optional: Filter dataset to only include valid classes? Or rely on sampler error handling.
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


    # --- Load Semantic Features ---
    semantic_feature_path = dynamic_paths['semantic_features']
    log.info(f"Loading semantic features from: {semantic_feature_path}")
    try:
        if not os.path.exists(semantic_feature_path):
             log.error(f"Semantic features file not found: {semantic_feature_path}"); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features_dict = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        # Features should already be normalized
        # semantic_features_dict = {k: normalize(v) for k, v in semantic_features_dict.items()}
        semantic_dim = config['model']['foundation_model']['text_encoder_dim'] # Use config dim
        # Verify loaded dim matches config dim
        loaded_dim = list(semantic_features_dict.values())[0].shape[-1]
        if loaded_dim != semantic_dim:
             log.warning(f"Loaded semantic feature dim ({loaded_dim}) != config dim ({semantic_dim}). Check config/generation.")
             semantic_dim = loaded_dim # Use loaded dim
        log.info(f"Loaded semantic features for {len(semantic_features_dict)} classes. Dim: {semantic_dim}")
    except Exception as e:
        log.error(f"Error loading semantic features: {e}", exc_info=True); sys.exit(1)

    # --- Load VLM (Visual Encoder - Frozen) ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path): log.error(f"VLM weights not found: {vlm_weights_path}"); sys.exit(1)
        log.info(f"Loading VLM weights from: {vlm_weights_path}")

        vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        visual_encoder = vlm_model.visual.to(device).eval()

        for param in visual_encoder.parameters(): param.requires_grad = False
        log.info("VLM Visual Encoder loaded and frozen.")

        expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                           (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
        if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]

        visual_dim = fm_config['visual_encoder_dim'] # Use config dim
        # Optionally verify VLM output dim matches config dim after first forward pass

    except Exception as e: log.error(f"Failed to load VLM: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained Adapter (Frozen) ---
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
        for param in adapter.parameters(): param.requires_grad = False
        log.info(f"Pre-trained adapter loaded and frozen (from epoch {adapter_checkpoint_data.get('epoch', 'N/A')}).")
    except Exception as e: log.error(f"Error loading adapter checkpoint: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained SemAlign Module (Frozen, Optional) ---
    semalign_module = None
    if any(k > 0 for k in kappas_to_test): # Only load if fusion is actually used
        log.info(f"Loading pre-trained SemAlign module from: {semalign_checkpoint_path}")
        try:
            if not os.path.exists(semalign_checkpoint_path): log.warning(f"SemAlign checkpoint not found: {semalign_checkpoint_path}. Fusion with kappa>0 might fail."); raise FileNotFoundError # Skip loading if not found
            semalign_module = SemAlignModule(
                visual_dim=visual_dim,
                semantic_dim=semantic_dim,
                hidden_dim=config['model']['fusion_module']['hidden_dim'],
                output_dim=visual_dim, # Output should match visual dim
                drop=0.0 # No dropout during testing
            ).to(device)
            semalign_checkpoint_data = torch.load(semalign_checkpoint_path, map_location=device)
            semalign_module.load_state_dict(semalign_checkpoint_data['semalign_state_dict'])
            semalign_module.eval()
            for param in semalign_module.parameters(): param.requires_grad = False
            log.info(f"Pre-trained SemAlign module loaded and frozen (from epoch {semalign_checkpoint_data.get('epoch', 'N/A')}).")
        except FileNotFoundError:
             semalign_module = None # Ensure it's None if loading failed
             log.warning("Proceeding without SemAlign module. Kappa values > 0 will effectively be kappa = 0.")
             kappas_to_test = [0.0] if 0.0 not in kappas_to_test else kappas_to_test # Force kappa=0 if module missing
        except Exception as e: log.error(f"Error loading SemAlign module checkpoint: {e}", exc_info=True); sys.exit(1) # Exit on other errors

    # --- Testing Loop ---
    log.info(f"Starting testing on {n_batch} episodes...")
    # Store results per kappa value across all episodes
    all_kappa_accuracies = {k: [] for k in kappas_to_test}

    pbar = tqdm(range(n_batch), desc="Testing Episodes")
    episode_indices_generator = iter(novel_sampler) # Use iterator for sampling

    for episode_idx in pbar:
        try:
            # --- Sample Episode Data ---
            batch_indices = next(episode_indices_generator) # Get indices for the episode
            support_indices = batch_indices[:n_way * k_shot]
            query_indices = batch_indices[n_way * k_shot:]

            # Get actual data and map labels correctly within the episode (0 to N-1)
            support_data_list = []
            support_labels_list = []
            query_data_list = []
            query_labels_list = []
            episode_class_names = [] # Original names of classes in this episode
            label_map = {} # Map original dataset label index to episode index (0 to N-1)

            current_episode_label = 0
            for i in range(n_way):
                cls_support_indices = support_indices[i*k_shot:(i+1)*k_shot]
                cls_query_indices = query_indices[i*q_query:(i+1)*q_query]

                # Get class name from the first support sample
                first_sample_idx = cls_support_indices[0].item()
                _, original_label_idx = novel_dataset[first_sample_idx]
                class_name = novel_dataset.idx_to_class[original_label_idx]
                episode_class_names.append(class_name)
                label_map[original_label_idx] = current_episode_label

                for idx in cls_support_indices:
                     data, _ = novel_dataset[idx.item()]
                     support_data_list.append(data)
                     support_labels_list.append(current_episode_label)
                for idx in cls_query_indices:
                     data, _ = novel_dataset[idx.item()]
                     query_data_list.append(data)
                     query_labels_list.append(current_episode_label)

                current_episode_label += 1

            support_data = torch.stack(support_data_list).to(device)
            query_data = torch.stack(query_data_list).to(device)
            # support_labels = torch.tensor(support_labels_list, device=device) # Not directly needed for proto calc
            query_labels_episode = torch.tensor(query_labels_list, device=device) # Ground truth for queries

            # --- Feature Extraction ---
            with torch.no_grad():
                # Support features (z_V)
                pseudo_images_support = adapter(support_data)
                if pseudo_images_support.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_support = F.interpolate(pseudo_images_support, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_support = visual_encoder(pseudo_images_support)
                if isinstance(z_v_support, tuple): z_v_support = z_v_support[0]
                if z_v_support.dim() == 3 and z_v_support.shape[1] > 1: z_v_support = z_v_support[:, 0]
                # Normalize support features before averaging for prototype
                z_v_support_norm = normalize(z_v_support)

                # Query features (z_V)
                pseudo_images_query = adapter(query_data)
                if pseudo_images_query.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_query = F.interpolate(pseudo_images_query, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_query = visual_encoder(pseudo_images_query)
                if isinstance(z_v_query, tuple): z_v_query = z_v_query[0]
                if z_v_query.dim() == 3 and z_v_query.shape[1] > 1: z_v_query = z_v_query[:, 0]
                z_v_query_norm = normalize(z_v_query) # Normalize query features once

                # Semantic features for the episode
                try:
                    # Order semantic features according to episode_class_names
                    z_t_episode = torch.stack([semantic_features_dict[name] for name in episode_class_names]).to(device)
                    # Already normalized during generation
                except KeyError as e:
                    log.warning(f"Episode {episode_idx}: Missing semantic feature for class {e}. Skipping episode.")
                    continue

                # Calculate mean visual prototype (u_c) using normalized support features
                z_v_support_norm_reshaped = z_v_support_norm.view(n_way, k_shot, visual_dim)
                u_c = z_v_support_norm_reshaped.mean(dim=1) # Shape: (N, D_vis)

                # Calculate reconstructed prototypes (r_c) if SemAlign module exists
                r_c = None
                if semalign_module is not None:
                    # Repeat semantics for each support sample per class
                    z_t_support_repeated = z_t_episode.repeat_interleave(k_shot, dim=0)
                    # Use UNNORMALIZED z_v_support as input to SemAlign module (matching SemFew training)
                    reconstructed_features = semalign_module(z_v_support, z_t_support_repeated)
                    # Normalize the output of SemAlign module before averaging
                    reconstructed_features_norm = normalize(reconstructed_features)
                    reconstructed_features_norm_reshaped = reconstructed_features_norm.view(n_way, k_shot, visual_dim)
                    r_c = reconstructed_features_norm_reshaped.mean(dim=1) # Shape: (N, D_vis)


            # --- Classification for each Kappa ---
            episode_accuracies = {}
            for current_kappa in kappas_to_test:
                with torch.no_grad():
                    # Determine prototype (p_c)
                    if current_kappa == 0 or r_c is None:
                        p_c = u_c # Use visual-only prototype
                        # u_c is already mean of normalized features, no need to normalize again
                    else:
                        # Fuse Prototypes: p_c = k * r_c + (1 - k) * u_c
                        # u_c and r_c are already means of normalized features
                        p_c = current_kappa * r_c + (1 - current_kappa) * u_c
                        # DO NOT NORMALIZE p_c here. Let Cosine_classifier handle it.
                        # p_c = normalize(p_c) # <--- REMOVED THIS LINE

                    # Classification
                    # Cosine_classifier normalizes p_c and z_v_query_norm internally
                    logits, predictions = Cosine_classifier(p_c, z_v_query_norm, temperature=classifier_temp)

                    # Calculate Accuracy
                    acc = (predictions == query_labels_episode).float().mean().item()
                    episode_accuracies[current_kappa] = acc
                    all_kappa_accuracies[current_kappa].append(acc)

            pbar.set_postfix({f'Acc(k={k:.1f})': f"{acc:.4f}" for k, acc in episode_accuracies.items()})

        except StopIteration:
             log.warning("Sampler exhausted before reaching n_batch episodes.")
             break # Exit loop if sampler finishes early
        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {e}", exc_info=True)
            continue # Skip to next episode

    # --- Aggregate and Report Results ---
    log.info(f"--- Testing Complete ({len(list(all_kappa_accuracies.values())[0])} episodes processed) ---")
    log.info(f"{n_way}-way {k_shot}-shot Results:")

    best_mean_acc = -1.0
    best_kappa = -1.0
    best_ci95 = 0.0

    results_summary = {}
    for k in kappas_to_test:
        accuracies_np = np.array(all_kappa_accuracies[k])
        if len(accuracies_np) == 0:
            log.warning(f"No results recorded for kappa={k}. Skipping.")
            continue
        mean_acc, ci95 = count_95acc(accuracies_np)
        log.info(f"  Kappa = {k:.2f}: Mean Accuracy = {mean_acc * 100:.2f}% +/- {ci95 * 100:.2f}%")
        results_summary[k] = {'mean': mean_acc, 'ci95': ci95}
        if mean_acc > best_mean_acc:
             best_mean_acc = mean_acc
             best_kappa = k
             best_ci95 = ci95

    log.info(f"--- Best Mean Accuracy across tested kappas ---")
    log.info(f"  Best Kappa: {best_kappa:.2f}")
    log.info(f"  Mean Accuracy: {best_mean_acc * 100:.2f}%")
    log.info(f"  95% CI: +/- {best_ci95 * 100:.2f}%")

    # Save results summary to file
    results_path = os.path.join(log_dir, 'results_summary.yaml')
    try:
        with open(results_path, 'w') as f:
            yaml.dump({
                'config_path': config_path,
                'vlm_variant': config['model']['foundation_model']['variant'],
                'text_type': config['semantics']['generation']['text_type'],
                'fsl_setting': f"{n_way}w{k_shot}s",
                'num_episodes': len(list(all_kappa_accuracies.values())[0]),
                'results_per_kappa': results_summary,
                'best_result': {'kappa': best_kappa, 'mean_acc': best_mean_acc, 'ci95': best_ci95}
            }, f, default_flow_style=False)
        log.info(f"Results summary saved to: {results_path}")
    except Exception as e:
        log.error(f"Failed to save results summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Learning Testing (Adapter Approach)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    parser.add_argument('--n_way', type=int, default=None, help='Override N-way (default: from config)')
    parser.add_argument('--k_shot', type=int, default=None, help='Override K-shot (default: from config)')
    parser.add_argument('--kappa', type=float, default=None, help='Test a specific kappa value (default: sweep from config or linspace)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific adapter checkpoint (default: latest from dynamic path)')
    args = parser.parse_args()

    main(args.config, args)