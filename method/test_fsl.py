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

# --- Added for t-SNE Visualization ---
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# -------------------------------------

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from data.samplers import CategoriesSampler
# Assuming you renamed HRRPGAFCNNAdapter to HRPPtoPseudoImage in the file
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage
from method.alignment import SemAlignModule
from logger import loggers
from utils import set_seed, normalize, Cosine_classifier, count_95acc, load_config, get_dynamic_paths # Import helpers

logging.basicConfig(level=logging.INFO) # Basic config for early messages
logger = logging.getLogger(__name__)

# Helper function to sample episode (Consistent with previous working version)
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
            # Ensure labels are sorted correctly if they are not contiguous 0, 1, 2...
            unique_dataset_labels = sorted(list(set(dataset.labels)))
            original_class_label_value = unique_dataset_labels[internal_cls_idx]
            class_name = dataset.idx_to_class[original_class_label_value] # Map label value to name
            episode_class_names.append(class_name)

            all_indices_for_class = sampler.m_ind[internal_cls_idx] # Use sampler's internal index
            if len(all_indices_for_class) < k_shot + q_query:
                # Handle insufficient samples - matching previous logic if needed
                logger.warning(f"Class {class_name} (Label {original_class_label_value}) has only {len(all_indices_for_class)} samples, needing {k_shot+q_query}. Skipping episode for safety.")
                # If replacement logic was intended, it should be implemented here. Sticking to skipping.
                return None, None, None
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
    q_query = config['fsl']['q_query']
    n_batch = config['fsl']['test_episodes']
    classifier_temp = config['fsl'].get('classifier_temperature', 10.0) # Using 10.0 based on original working code

    # Kappa handling
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
    adapter_checkpoint_path = args_override.checkpoint if args_override.checkpoint else dynamic_paths['adapter_latest_ckpt']
    semalign_checkpoint_path = dynamic_paths['semalign_latest_ckpt'] # SemAlign checkpoint should correspond to the adapter

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
    log.info(f"Logs and plots will be saved to: {log_dir}") # Updated log message

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

        # Check sample counts
        samples_per_class = {}
        for i in range(len(novel_dataset)):
             _, label_idx = novel_dataset[i]
             label_name = novel_dataset.idx_to_class[label_idx]
             samples_per_class[label_name] = samples_per_class.get(label_name, 0) + 1
        min_samples = k_shot + q_query
        valid_classes = [cls for cls, count in samples_per_class.items() if count >= min_samples]
        if len(valid_classes) < n_way:
            log.error(f"Not enough classes ({len(valid_classes)}) with sufficient samples ({min_samples}) for {n_way}-way {k_shot}-shot {q_query}-query evaluation.")
            log.error(f"Classes with counts: {samples_per_class}"); sys.exit(1)
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
        if not os.path.exists(semantic_feature_path): log.error(f"Semantic features file not found: {semantic_feature_path}"); sys.exit(1)
        semantic_data = torch.load(semantic_feature_path, map_location='cpu')
        semantic_features_dict = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items()}
        # Features should already be normalized from generation script
        semantic_dim = config['model']['foundation_model']['text_encoder_dim']
        loaded_dim = list(semantic_features_dict.values())[0].shape[-1]
        if loaded_dim != semantic_dim:
            log.warning(f"Loaded semantic dim {loaded_dim} != config dim {semantic_dim}")
            semantic_dim = loaded_dim # Adjust if needed
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
        visual_dim = fm_config['visual_encoder_dim']
    except Exception as e: log.error(f"Failed to load VLM: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained Adapter (Frozen) ---
    log.info(f"Loading pre-trained adapter from: {adapter_checkpoint_path}")
    try:
        if not os.path.exists(adapter_checkpoint_path): log.error(f"Adapter checkpoint not found: {adapter_checkpoint_path}"); sys.exit(1)

        # ***** MODIFIED SECTION *****
        # Read GAF+CNN adapter config from the YAML file
        adapter_gaf_config = config['model'].get('adapter_gaf_cnn')
        if adapter_gaf_config is None:
            raise ValueError("Config section 'model.adapter_gaf_cnn' is missing in the YAML file.")

        # Instantiate HRPPtoPseudoImage (containing HRRPGAFCNNAdapter code)
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'],
            input_channels=1, # Assuming magnitude input
            gaf_size=adapter_gaf_config.get('gaf_size', 64),
            cnn_channels=adapter_gaf_config.get('cnn_channels', [16, 32, 64]),
            output_channels=3, # Target for CLIP visual encoder
            output_size=expected_img_size, # Target for CLIP visual encoder
            kernel_size=adapter_gaf_config.get('cnn_kernel_size', 3),
            activation=adapter_gaf_config.get('cnn_activation', 'relu'),
            use_batchnorm=adapter_gaf_config.get('cnn_use_batchnorm', True)
        ).to(device)
        # ***** END MODIFIED SECTION *****

        adapter_checkpoint_data = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint_data['adapter_state_dict'])
        adapter.eval()
        for param in adapter.parameters(): param.requires_grad = False
        log.info(f"Pre-trained adapter loaded and frozen (from epoch {adapter_checkpoint_data.get('epoch', 'N/A')}).")

    except KeyError as e:
        log.error(f"Missing key when loading/instantiating adapter or its config: {e}. Please check your YAML file and checkpoint.")
        sys.exit(1)
    except ValueError as e:
        log.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e: log.error(f"Error loading adapter checkpoint: {e}", exc_info=True); sys.exit(1)

    # --- Load Pre-trained SemAlign Module (Frozen, Optional) ---
    semalign_module = None
    if any(k > 0 for k in kappas_to_test):
        log.info(f"Loading pre-trained SemAlign module from: {semalign_checkpoint_path}")
        try:
            if not os.path.exists(semalign_checkpoint_path):
                log.warning(f"SemAlign checkpoint not found: {semalign_checkpoint_path}. Fusion with kappa>0 will fail.");
                raise FileNotFoundError("SemAlign checkpoint missing")

            # Determine visual_dim again just before loading SemAlign
            actual_visual_dim = fm_config['visual_encoder_dim'] # Use config value directly

            semalign_module = SemAlignModule(
                visual_dim=actual_visual_dim, # Use actual/expected visual dim
                semantic_dim=semantic_dim,
                hidden_dim=config['model']['fusion_module']['hidden_dim'],
                output_dim=actual_visual_dim, # Output dim must match visual dim
                drop=0.0 # No dropout during inference
            ).to(device)
            semalign_checkpoint_data = torch.load(semalign_checkpoint_path, map_location=device)
            semalign_module.load_state_dict(semalign_checkpoint_data['semalign_state_dict'])
            semalign_module.eval()
            for param in semalign_module.parameters(): param.requires_grad = False
            log.info(f"Pre-trained SemAlign module loaded and frozen (from epoch {semalign_checkpoint_data.get('epoch', 'N/A')}).")
        except FileNotFoundError:
            semalign_module = None
            log.warning("Proceeding without SemAlign module. Kappa values > 0 will be ineffective.")
            kappas_to_test = [k for k in kappas_to_test if k == 0.0] # Only test kappa=0 if module missing
            if not kappas_to_test: # Add kappa=0 if it wasn't there initially
                 kappas_to_test = [0.0]
                 log.info("Added kappa=0.0 to tests as SemAlign module is missing.")

        except Exception as e: log.error(f"Error loading SemAlign module checkpoint: {e}", exc_info=True); sys.exit(1)

    # --- Testing Loop ---
    log.info(f"Starting testing on {n_batch} episodes...")
    all_kappa_accuracies = {k: [] for k in kappas_to_test}
    pbar = tqdm(range(n_batch), desc="Testing Episodes")
    episode_indices_generator = iter(novel_sampler)
    tsne_plot_done = False # Flag to plot only the first episode

    for episode_idx in pbar:
        try:
            # --- Sample Episode Data ---
            # Use the helper function for consistency
            support_indices, query_indices, episode_class_names = sample_one_episode(novel_dataset, novel_sampler, n_way, k_shot, q_query)
            if support_indices is None:
                log.warning(f"Could not sample episode {episode_idx}. Skipping.")
                continue

            support_data = torch.stack([novel_dataset[i][0] for i in support_indices]).to(device)
            query_data = torch.stack([novel_dataset[i][0] for i in query_indices]).to(device)
            # Create query labels: [0, 0, ..., 1, 1, ..., N-1, N-1, ...]
            query_labels_episode = torch.arange(n_way, device=device).repeat_interleave(q_query)

            # --- Feature Extraction ---
            with torch.no_grad():
                # Support features (z_v_support) - UNNORMALIZED
                pseudo_images_support = adapter(support_data)
                if pseudo_images_support.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_support = F.interpolate(pseudo_images_support, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_support = visual_encoder(pseudo_images_support)
                if isinstance(z_v_support, tuple): z_v_support = z_v_support[0]
                if z_v_support.dim() == 3 and z_v_support.shape[1] > 1: z_v_support = z_v_support[:, 0]

                # Query features (z_v_query) - UNNORMALIZED initially
                pseudo_images_query = adapter(query_data)
                if pseudo_images_query.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_query = F.interpolate(pseudo_images_query, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                z_v_query = visual_encoder(pseudo_images_query)
                if isinstance(z_v_query, tuple): z_v_query = z_v_query[0]
                if z_v_query.dim() == 3 and z_v_query.shape[1] > 1: z_v_query = z_v_query[:, 0]
                # Normalize query features ONCE for classification
                z_v_query_norm = normalize(z_v_query)

                # Semantic features for the episode (z_t_episode) - ALREADY NORMALIZED
                try:
                    z_t_episode = torch.stack([semantic_features_dict[name] for name in episode_class_names]).to(device)
                except KeyError as e:
                    log.warning(f"Episode {episode_idx}: Missing semantic feature for class {e}. Skipping episode."); continue

                # Calculate u_t (Mean Visual Prototype) - Mean then Normalize (Original ProtoNet way)
                z_v_support_reshaped = z_v_support.view(n_way, k_shot, visual_dim)
                u_t = z_v_support_reshaped.mean(dim=1) # (N, D_vis) - UNNORMALIZED mean
                u_t_norm = normalize(u_t) # NORMALIZE the mean prototype

                # Calculate r_t (Reconstructed Prototype) - Mean then Normalize (Corrected SemFew way)
                r_t = None
                r_t_norm = None
                if semalign_module is not None:
                    z_t_support_repeated = z_t_episode.repeat_interleave(k_shot, dim=0)
                    # Input UNNORMALIZED z_v_support to SemAlign
                    reconstructed_features = semalign_module(z_v_support, z_t_support_repeated) # (N*K, D_vis) - UNNORMALIZED
                    reconstructed_features_reshaped = reconstructed_features.view(n_way, k_shot, visual_dim)
                    r_t = reconstructed_features_reshaped.mean(dim=1) # (N, D_vis) - UNNORMALIZED mean
                    r_t_norm = normalize(r_t) # NORMALIZE the mean reconstructed prototype


            # --- [START] t-SNE Visualization (for the first episode) ---
            if not tsne_plot_done:
                log.info(f"Generating t-SNE plot for episode {episode_idx}...")
                try:
                    # Prepare data for t-SNE
                    support_labels_episode = torch.arange(n_way, device=device).repeat_interleave(k_shot) # Create support labels
                    all_features_episode = torch.cat((z_v_support, z_v_query), dim=0)
                    all_labels_episode = torch.cat((support_labels_episode, query_labels_episode), dim=0)
                    is_query_indicator = torch.cat((torch.zeros(len(support_labels_episode)), torch.ones(len(query_labels_episode))), dim=0)

                    num_total_samples = all_features_episode.shape[0]
                    if num_total_samples < 3: # t-SNE needs at least 3 samples
                         log.warning(f"Skipping t-SNE for episode {episode_idx}: Not enough samples ({num_total_samples}).")
                    else:
                        # Ensure perplexity is valid
                        perplexity_value = min(30.0, num_total_samples - 1.0)

                        tsne = TSNE(n_components=2,
                                    random_state=config.get('seed', 42),
                                    perplexity=perplexity_value,
                                    learning_rate='auto',
                                    init='pca',
                                    n_iter=1000)

                        features_np = all_features_episode.detach().cpu().numpy()
                        labels_np = all_labels_episode.detach().cpu().numpy()
                        is_query_np = is_query_indicator.detach().cpu().numpy()

                        tsne_results = tsne.fit_transform(features_np)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(12, 10))
                        cmap = plt.cm.get_cmap("tab10", n_way) # Colormap for N classes

                        for i in range(n_way):
                            # Support samples for class i
                            support_mask = (labels_np == i) & (is_query_np == 0)
                            ax.scatter(tsne_results[support_mask, 0], tsne_results[support_mask, 1],
                                       color=cmap(i), marker='o', s=60, alpha=0.9,
                                       label=episode_class_names[i] if episode_idx == 0 else "") # Label only once

                            # Query samples for class i
                            query_mask = (labels_np == i) & (is_query_np == 1)
                            ax.scatter(tsne_results[query_mask, 0], tsne_results[query_mask, 1],
                                       marker='o', s=60, alpha=0.9,
                                       facecolors='none', edgecolors=cmap(i), linewidths=1.5)
                                       # label=f"{episode_class_names[i]} (Query)") # Avoid double labeling legend

                        # Create custom legend handles for Support/Query distinction
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], marker='o', color='gray', label='Support', linestyle='None', markersize=8),
                            Line2D([0], [0], marker='o', color='gray', label='Query', linestyle='None', markersize=8, markerfacecolor='none', markeredgewidth=1.5)
                        ]
                        # Add class name handles
                        for i in range(n_way):
                            legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=episode_class_names[i]))


                        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
                        ax.set_title(f"t-SNE Visualisation of Visual Features (Episode {episode_idx}, {n_way}-way {k_shot}-shot)")
                        ax.set_xlabel("t-SNE Component 1")
                        ax.set_ylabel("t-SNE Component 2")
                        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

                        tsne_save_path = os.path.join(log_dir, f'tsne_episode_{episode_idx}.pdf')
                        plt.savefig(tsne_save_path, bbox_inches='tight')
                        plt.close(fig)
                        log.info(f"t-SNE plot saved to {tsne_save_path}")
                        tsne_plot_done = True # Ensure plot is done only once

                except Exception as plot_err:
                    log.error(f"Failed to generate or save t-SNE plot for episode {episode_idx}: {plot_err}", exc_info=True)
                    # Don't set tsne_plot_done = True, maybe try next episode if this one failed?
                    # Or just accept it failed for this run. Let's accept failure and move on.
                    tsne_plot_done = True # Mark as 'attempted' to avoid repeated errors

            # --- [END] t-SNE Visualization ---


            # --- Classification for each Kappa ---
            episode_accuracies = {}
            for current_kappa in kappas_to_test:
                with torch.no_grad():
                    # Determine prototype (p_t)
                    if current_kappa == 0 or r_t_norm is None:
                        # Use visual-only normalized prototype
                        p_t_for_classification = u_t_norm
                    else:
                        # Fuse NORMALIZED prototypes: p_t = k * r_t_norm + (1 - k) * u_t_norm
                        # Note: This linear combination of normalized vectors might not be unit norm.
                        p_t_combined = current_kappa * r_t_norm + (1 - current_kappa) * u_t_norm
                        # Let Cosine_classifier handle the final normalization of p_t_combined
                        p_t_for_classification = p_t_combined

                    # Classification
                    # Cosine_classifier normalizes p_t_for_classification and z_v_query_norm internally
                    logits, predictions = Cosine_classifier(p_t_for_classification, z_v_query_norm, temperature=classifier_temp)

                    # Calculate Accuracy
                    acc = (predictions == query_labels_episode).float().mean().item()
                    episode_accuracies[current_kappa] = acc
                    all_kappa_accuracies[current_kappa].append(acc)

            # Find best kappa and accuracy for this episode for logging postfix
            best_acc_this_episode = max(episode_accuracies.values()) if episode_accuracies else -1.0
            best_kappa_this_episode = kappas_to_test[np.argmax(list(episode_accuracies.values()))] if episode_accuracies else -1.0
            pbar.set_postfix({'Vis Acc(k=0)': f"{episode_accuracies.get(0.0, -1):.4f}", 'Best Fused': f"{best_acc_this_episode:.4f}", 'Best k': f"{best_kappa_this_episode:.2f}"})


        except StopIteration:
             log.warning("Sampler exhausted before reaching n_batch episodes."); break
        except Exception as e:
            log.error(f"Error processing episode {episode_idx}: {e}", exc_info=True); continue

    # --- Aggregate and Report Results ---
    num_processed = len(list(all_kappa_accuracies.values())[0]) if all_kappa_accuracies and all_kappa_accuracies.values() and len(list(all_kappa_accuracies.values())[0]) > 0 else 0
    if num_processed == 0:
         log.error("No episodes were successfully processed. Cannot calculate results.")
         sys.exit(1) # Exit if no results

    log.info(f"--- Testing Complete ({num_processed} episodes processed) ---")
    log.info(f"{n_way}-way {k_shot}-shot Results:")

    best_mean_acc = -1.0
    best_kappa = -1.0
    best_ci95 = 0.0
    results_summary = {}

    # Report visual-only (kappa=0) baseline separately
    if 0.0 in all_kappa_accuracies:
        visual_only_acc_np = np.array(all_kappa_accuracies[0.0])
        if len(visual_only_acc_np) > 0:
            mean_visual_only, ci95_visual_only = count_95acc(visual_only_acc_np)
            log.info(f"  Visual-Only (Kappa = 0.0): Mean Acc = {mean_visual_only * 100:.2f}% +/- {ci95_visual_only * 100:.2f}%")
            results_summary[0.0] = {'mean': mean_visual_only, 'ci95': ci95_visual_only}
            if mean_visual_only > best_mean_acc:
                 best_mean_acc, best_kappa, best_ci95 = mean_visual_only, 0.0, ci95_visual_only
        else:
            log.warning("No results recorded for kappa=0.0.")
    else:
        log.warning("Kappa=0.0 was not tested.")

    # Report results for other kappas and find overall best
    if any(k > 0 for k in kappas_to_test): # Only print header if fusion was tested
        log.info(f"  Fused Prototypes (Kappa > 0):")
    for k in kappas_to_test:
        if k == 0.0: continue # Skip kappa=0 as it's reported above
        accuracies_np = np.array(all_kappa_accuracies[k])
        if len(accuracies_np) == 0:
            log.warning(f"No results recorded for kappa={k}. Skipping.")
            continue
        mean_acc, ci95 = count_95acc(accuracies_np)
        log.info(f"    Kappa = {k:.2f}: Mean Acc = {mean_acc * 100:.2f}% +/- {ci95 * 100:.2f}%")
        results_summary[k] = {'mean': mean_acc, 'ci95': ci95}
        if mean_acc > best_mean_acc:
             best_mean_acc = mean_acc
             best_kappa = k
             best_ci95 = ci95

    log.info(f"--- Overall Best Mean Accuracy Across Tested Kappas ---")
    log.info(f"  Best Kappa: {best_kappa:.2f}")
    log.info(f"  Mean Accuracy: {best_mean_acc * 100:.2f}%")
    log.info(f"  95% CI: +/- {best_ci95 * 100:.2f}%")

    # Save results summary
    results_path = os.path.join(log_dir, 'results_summary.yaml')
    try:
        # Convert float kappas to strings for YAML keys if necessary, although float keys often work
        results_summary_yaml = {str(k): v for k, v in results_summary.items()}
        best_result_yaml = {'kappa': float(best_kappa), 'mean_acc': float(best_mean_acc), 'ci95': float(best_ci95)} # Ensure best results are standard floats

        with open(results_path, 'w') as f:
            yaml.dump({
                'config_path': config_path,
                'vlm_variant': config['model']['foundation_model']['variant'],
                'text_type': config['semantics']['generation']['text_type'],
                'fsl_setting': f"{n_way}w{k_shot}s",
                'num_episodes': num_processed,
                'results_per_kappa': results_summary_yaml, # Use YAML-safe keys
                'best_result': best_result_yaml
            }, f, default_flow_style=False, sort_keys=False) # Keep order if possible
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