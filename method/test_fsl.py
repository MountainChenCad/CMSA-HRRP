import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F # Added
from torch.utils.data import DataLoader
import open_clip # Assuming RemoteCLIP

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.hrrp_dataset import HRRPDataset
from data.samplers import CategoriesSampler
# from model.hrrp_encoder import HRRPEncoder # No longer needed
from model.hrrp_adapter_1d_to_2d import HRPPtoPseudoImage # Import the adapter
# from method.alignment import AlignmentModule # No longer needed
from method.alignment import FusionModule # Keep FusionModule definition
from logger import loggers
from utils import set_seed, Cosine_classifier, count_95acc, normalize # Need normalize

# Configure root logger basic settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
# Create a specific logger instance for this module
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f: # Added encoding
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config, args_override):
    """Main Few-Shot Learning evaluation function using Adapter + VLM Visual Encoder."""

    # --- Setup ---
    n_way = args_override.n_way if args_override.n_way is not None else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot is not None else config['fsl']['k_shot']
    q_query = args_override.q_query if args_override.q_query is not None else config['fsl']['q_query']
    test_episodes = args_override.test_episodes if args_override.test_episodes is not None else config['fsl']['test_episodes']
    kappa = args_override.kappa if args_override.kappa is not None else config['model']['fusion_module']['kappa']
    # Load adapter checkpoint, not alignment module checkpoint
    adapter_checkpoint_path = args_override.checkpoint if args_override.checkpoint else os.path.join(config['paths']['checkpoints'], 'hrrp_adapter', 'latest.pth') # Changed path

    # Determine if fusion is used
    use_fusion = kappa > 0

    # Setup Log directory based on test settings
    setting_name = f"Adapter_{n_way}way_{k_shot}shot_k{kappa:.2f}" # Reflect new approach
    log_dir = os.path.join(config['paths'].get('logs', './logs'), 'fsl_testing_adapter', setting_name) # New log subdir
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, 'test_fsl_adapter.log')
    loggers(log_file_path) # Configure root logger
    logger.info(f"Logger configured. Logging to: {log_file_path}")

    logger.info(f"Loaded base configuration: {config}")
    logger.info(f"Testing with Overrides: N-Way={n_way}, K-Shot={k_shot}, Q-Query={q_query}, Episodes={test_episodes}, Kappa={kappa}, Ckpt={adapter_checkpoint_path}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading (Novel Set) ---
    logger.info("Loading novel dataset...")
    try:
        novel_dataset = HRRPDataset(
            root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
            split='novel',
            classes=config['data']['novel_classes'],
            target_length=config['data']['target_length'],
            normalization=config['data']['normalization'],
            phase_info='magnitude' # Must match adapter input expectation
        )
    except Exception as e:
        logger.error(f"Failed to initialize Novel HRRPDataset: {e}")
        sys.exit(1)

    if len(novel_dataset.target_classes) < n_way:
         logger.error(f"Not enough novel classes ({len(novel_dataset.target_classes)}) for {n_way}-way setting.")
         sys.exit(1)
    min_samples_per_class = float('inf')
    for cls_name in novel_dataset.target_classes:
        cls_indices = [i for i, label in enumerate(novel_dataset.labels) if novel_dataset.idx_to_class[label] == cls_name]
        min_samples_per_class = min(min_samples_per_class, len(cls_indices))
    required_samples_per_class = k_shot + q_query
    if min_samples_per_class < required_samples_per_class:
        logger.error(f"Not enough samples per novel class. Required: {required_samples_per_class}, Minimum found: {min_samples_per_class}.")
        sys.exit(1)

    novel_sampler = CategoriesSampler(
        novel_dataset.labels,
        n_batch=test_episodes,
        n_cls=n_way,
        n_per=required_samples_per_class
    )
    num_workers_override = config.get('num_workers', 0)
    logger.info(f"Using num_workers = {num_workers_override}")
    novel_loader = DataLoader(
        novel_dataset,
        batch_sampler=novel_sampler,
        num_workers=num_workers_override,
        pin_memory=True
    )
    logger.info(f"Novel dataset loaded with {len(novel_dataset)} samples across {len(novel_dataset.target_classes)} classes.")
    logger.info(f"Sampling {test_episodes} episodes: {n_way}-way, {k_shot}-shot, {q_query}-query.")

    # --- Load Semantic Features (Novel Classes) ---
    logger.info(f"Loading semantic features from: {config['semantics']['feature_path']}")
    try:
        semantic_data = torch.load(config['semantics']['feature_path'], map_location='cpu')
        semantic_features = {k: v.float().to(device)
                             for k, v in semantic_data['semantic_feature'].items()
                             if k in config['data']['novel_classes']}
        if len(semantic_features) != len(config['data']['novel_classes']):
             logger.warning(f"Mismatch between loaded semantic features ({len(semantic_features)}) and configured novel classes ({len(config['data']['novel_classes'])}).")
        logger.info(f"Loaded semantic features for {len(semantic_features)} novel classes.")
    except Exception as e:
         logger.error(f"Error loading semantic features: {e}")
         sys.exit(1)

    # --- Load VLM (Visual Encoder) ---
    fm_config = config['model']['foundation_model']
    logger.info(f"Loading VLM Visual Encoder: {fm_config['name']} ({fm_config['variant']})")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            fm_variant = fm_config['variant']
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"
            local_weights_path = os.path.join(base_checkpoint_dir, 'foundation_models', weights_filename)
            if not os.path.exists(local_weights_path):
                 logger.error(f"VLM weights file not found: {local_weights_path}")
                 sys.exit(1)
            logger.info(f"Loading VLM weights from: {local_weights_path}")

            vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_variant, pretrained=local_weights_path)
            visual_encoder = vlm_model.visual.to(device).eval() # f_V
            for param in visual_encoder.parameters():
                param.requires_grad = False
            logger.info("VLM Visual Encoder loaded and frozen.")
            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
            if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
        else:
            logger.error(f"Unsupported foundation model: {fm_config['name']}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        sys.exit(1)

    # --- Adapter Initialization and Loading Weights ---
    try:
        hrrp_adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'],
            input_channels=1, # Assuming magnitude
            output_channels=3,
            output_size=expected_img_size,
            # Add other adapter params from config if defined
        ).to(device)

        logger.info(f"Loading adapter checkpoint from: {adapter_checkpoint_path}")
        adapter_checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        hrrp_adapter.load_state_dict(adapter_checkpoint['adapter_state_dict'])
        logger.info(f"Adapter weights loaded successfully from epoch {adapter_checkpoint.get('epoch', 'N/A')}.")
        hrrp_adapter.eval()

    except FileNotFoundError:
        logger.error(f"Adapter checkpoint file not found at {adapter_checkpoint_path}.")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing key 'adapter_state_dict' in checkpoint file {adapter_checkpoint_path}: {e}.")
        sys.exit(1)
    except Exception as e:
         logger.error(f"Error initializing or loading adapter: {e}")
         sys.exit(1)

    # --- Fusion Module Initialization and Loading (Optional) ---
    fusion_module = None
    if use_fusion:
        try:
            # Input dim is visual_dim + semantic_dim, Output dim is visual_dim
            visual_dim = fm_config['text_encoder_dim'] # CLIP features are usually same dim
            semantic_dim = fm_config['text_encoder_dim']
            fusion_module = FusionModule(
                aligned_hrrp_dim=visual_dim, # Input 1 is now visual feature
                semantic_dim=semantic_dim,   # Input 2 is semantic feature
                hidden_dim=config['model']['fusion_module']['hidden_dim'],
                output_dim=visual_dim        # Output should be in visual space
            ).to(device)

            # Load weights if they exist (e.g., from adapter checkpoint or separate file)
            if 'fusion_module_state_dict' in adapter_checkpoint: # Check if saved with adapter
                fusion_module.load_state_dict(adapter_checkpoint['fusion_module_state_dict'])
                logger.info("Fusion module weights loaded from adapter checkpoint.")
            else:
                 # Try loading from alignment checkpoint as fallback (old structure)
                 try:
                     old_ckpt_path = os.path.join(config['paths']['checkpoints'], 'alignment_module', 'latest.pth')
                     old_ckpt = torch.load(old_ckpt_path, map_location=device)
                     if 'fusion_module_state_dict' in old_ckpt:
                         fusion_module.load_state_dict(old_ckpt['fusion_module_state_dict'])
                         logger.info("Fusion module weights loaded from older alignment checkpoint.")
                     else:
                          logger.warning("Fusion module state dict not found in any checkpoint, but kappa > 0. Using initial weights.")
                 except FileNotFoundError:
                     logger.warning("Fusion module state dict not found and older alignment checkpoint missing. Using initial weights.")

            fusion_module.eval()
        except Exception as e:
            logger.error(f"Error initializing or loading fusion module: {e}. Setting kappa=0.")
            kappa = 0 # Disable fusion if module fails
            use_fusion = False


    # --- Evaluation Loop ---
    logger.info("Starting few-shot evaluation with Adapter...")
    all_accuracies = []
    query_labels_proto = torch.arange(n_way).repeat_interleave(q_query).long().to(device)

    pbar = tqdm(novel_loader, total=test_episodes, desc="Evaluating Episodes")
    for episode_idx, batch_data in enumerate(pbar):
        if episode_idx >= test_episodes: break

        try:
            # Data loading from batch_data (yields data, labels)
            hrrp_data_all, labels_all_original_tensor = batch_data
            hrrp_data_all = hrrp_data_all.to(device)
            labels_all_original = labels_all_original_tensor.tolist()

            # Map labels
            support_original_labels = labels_all_original[:n_way * k_shot]
            unique_labels_in_episode = sorted(list(set(support_original_labels)))
            if len(unique_labels_in_episode) < n_way:
                 logger.warning(f"Episode {episode_idx}: Expected {n_way} classes, found {len(unique_labels_in_episode)}. Skipping.")
                 continue
            label_map = {orig_label: episode_label for episode_label, orig_label in enumerate(unique_labels_in_episode)}
            episode_labels_all = torch.tensor([label_map[l] for l in labels_all_original]).to(device)

        except Exception as e:
            logger.error(f"Error processing batch data for episode {episode_idx}: {e}")
            continue

        # --- Feature Extraction using Adapter + Visual Encoder ---
        with torch.no_grad():
            try:
                pseudo_images_all = hrrp_adapter(hrrp_data_all) # (B, C, H, W)
                # Optional resize if needed
                if pseudo_images_all.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_all = F.interpolate(pseudo_images_all, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)

                visual_features_all = visual_encoder(pseudo_images_all) # z_V = f_V(h_1D_to_2D(x_H))
                if isinstance(visual_features_all, tuple): visual_features_all = visual_features_all[0]
                if visual_features_all.dim() == 3 and visual_features_all.shape[1] > 1:
                     visual_features_all = visual_features_all[:, 0] # Take CLS token for ViT

                visual_features_all = normalize(visual_features_all) # Normalize visual features

                if visual_features_all.dim() != 2:
                     logger.error(f"Visual features have unexpected shape: {visual_features_all.shape}. Expected 2D. Skipping.")
                     continue

            except Exception as e:
                 logger.error(f"Error during feature extraction (adapter/VLM) for episode {episode_idx}: {e}")
                 continue

        # Split support and query (now using VISUAL features z_V)
        support_features = visual_features_all[:n_way * k_shot]
        query_features = visual_features_all[n_way * k_shot:]
        support_labels_episode = episode_labels_all[:n_way * k_shot]

        # --- Calculate Prototypes (using VISUAL features) ---
        prototypes = []
        prototype_calculation_failed = False
        with torch.no_grad():
            for c in range(n_way):
                class_mask = (support_labels_episode == c)
                if not torch.any(class_mask):
                     logger.warning(f"Episode {episode_idx}: No support samples for class {c}. Skipping.")
                     prototype_calculation_failed = True; break
                class_support_features = support_features[class_mask] # (K_shot, visual_dim)

                original_label = unique_labels_in_episode[c]
                class_name = novel_dataset.idx_to_class[original_label]

                # Mean VISUAL feature (u_c)
                mean_visual_feature = class_support_features.mean(dim=0).squeeze() # (visual_dim,)
                if mean_visual_feature.dim() != 1:
                     logger.error(f"Ep {episode_idx}, Cls {c}: Mean visual feature not 1D ({mean_visual_feature.shape}). Skip.")
                     prototype_calculation_failed = True; break

                if use_fusion and fusion_module is not None: # Check if fusion is enabled AND module exists
                    if class_name in semantic_features:
                        class_semantic_feature = semantic_features[class_name].squeeze() # (semantic_dim,)
                        if class_semantic_feature.dim() != 1:
                             logger.error(f"Ep {episode_idx}, Cls {c}: Semantic feature not 1D ({class_semantic_feature.shape}). Using kappa=0 fallback.")
                             final_prototype = mean_visual_feature
                        else:
                             # Ensure dimensions match for fusion module input (visual_dim, semantic_dim)
                             if class_support_features.shape[-1] != visual_dim or \
                                class_semantic_feature.shape[-1] != semantic_dim:
                                 logger.error(f"Ep {episode_idx}, Cls {c}: Dim mismatch for fusion. Support={class_support_features.shape[-1]} vs {visual_dim}, Sem={class_semantic_feature.shape[-1]} vs {semantic_dim}. Using kappa=0 fallback.")
                                 final_prototype = mean_visual_feature
                             else:
                                 repeated_semantics = class_semantic_feature.unsqueeze(0).repeat(class_support_features.size(0), 1)
                                 # Fusion Module now takes visual features and semantic features
                                 reconstructed = fusion_module(class_support_features, repeated_semantics) # Output: (K_shot, visual_dim)
                                 mean_reconstructed = reconstructed.mean(dim=0).squeeze() # (visual_dim,)

                                 if mean_reconstructed.dim() != 1:
                                     logger.error(f"Ep {episode_idx}, Cls {c}: Mean reconstructed not 1D ({mean_reconstructed.shape}). Using kappa=0 fallback.")
                                     final_prototype = mean_visual_feature
                                 else:
                                     # Combine mean visual feature and reconstructed feature
                                     try:
                                         if mean_visual_feature.shape[0] != visual_dim or mean_reconstructed.shape[0] != visual_dim:
                                              logger.error(f"Ep {episode_idx}, Cls {c}: Dim mismatch in final fusion. Mean={mean_visual_feature.shape[0]}, Recon={mean_reconstructed.shape[0]}, Expected={visual_dim}. Using kappa=0 fallback.")
                                              final_prototype = mean_visual_feature
                                         else:
                                             final_prototype = float(kappa) * mean_reconstructed + (1.0 - float(kappa)) * mean_visual_feature
                                     except RuntimeError as e:
                                         logger.error(f"RUNTIME ERROR during prototype fusion for class {c}: {e}")
                                         final_prototype = mean_visual_feature # Fallback
                    else:
                        logger.warning(f"Ep {episode_idx}: Semantic feature for '{class_name}' missing. Using kappa=0.")
                        final_prototype = mean_visual_feature
                else:
                     # Fusion not used (kappa=0 or module failed)
                     final_prototype = mean_visual_feature

                # Ensure final prototype is 1D
                if final_prototype.dim() != 1:
                     logger.warning(f"Ep {episode_idx}, Cls {c}: Final proto not 1D ({final_prototype.shape}). Squeezing.")
                     final_prototype = final_prototype.squeeze()
                     if final_prototype.dim() != 1:
                         logger.error(f"Ep {episode_idx}, Cls {c}: Cannot make final proto 1D ({final_prototype.shape}). Skipping.")
                         prototype_calculation_failed = True; break

                prototypes.append(final_prototype)

        if prototype_calculation_failed: continue

        # --- Stack Prototypes and Classify ---
        try:
            prototypes = torch.stack(prototypes) # Shape: (n_way, visual_dim)
            if prototypes.dim() != 2 or prototypes.size(0) != n_way or prototypes.size(1) != visual_dim:
                 logger.error(f"Stacked prototypes shape error: {prototypes.shape}. Expected ({n_way}, {visual_dim}). Skipping.")
                 continue
        except Exception as e:
            logger.error(f"Error stacking prototypes: {e}. Shapes: {[p.shape for p in prototypes]}. Skipping.")
            continue

        # Classify Query Set (using VISUAL features)
        with torch.no_grad():
            if query_features.shape[-1] != visual_dim:
                 logger.error(f"Query features dim ({query_features.shape[-1]}) != prototype dim ({visual_dim}). Skipping.")
                 continue

            logits, predictions = Cosine_classifier(prototypes, query_features)
            accuracy = (predictions == query_labels_proto).float().mean().item()
            all_accuracies.append(accuracy)

        pbar.set_postfix({'Acc': f"{accuracy*100:.2f}%"})


    # --- Final Results ---
    if not all_accuracies:
         logger.error("No episodes were successfully evaluated.")
         return

    mean_acc, conf_interval = count_95acc(np.array(all_accuracies))
    logger.info(f"--- Final Results ({setting_name}) ---")
    logger.info(f"Evaluated {len(all_accuracies)} episodes.")
    logger.info(f"Average Accuracy: {mean_acc * 100:.2f}%")
    logger.info(f"95% Confidence Interval: {conf_interval * 100:.2f}%")
    logger.info(f"Result: {mean_acc * 100:.2f} ± {conf_interval * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot HRRP Recognition (Adapter + VLM Visual)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the trained ADAPTER checkpoint (overrides config)') # Changed help text
    parser.add_argument('--n_way', type=int, default=None, help='N-way (overrides config)')
    parser.add_argument('--k_shot', type=int, default=None, help='K-shot (overrides config)')
    parser.add_argument('--q_query', type=int, default=None, help='Query samples per class (overrides config)')
    parser.add_argument('--test_episodes', type=int, default=None, help='Number of test episodes (overrides config)')
    parser.add_argument('--kappa', type=float, default=None, help='Prototype fusion factor kappa (overrides config, >0 enables fusion)')
    # --no_align is no longer relevant for this approach
    # parser.add_argument('--no_align', action='store_true', help='Run the NoAlign baseline (bypass alignment module)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logging.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    # Remove --no_align logic if it was passed but irrelevant now
    args.no_align = False
    main(configuration, args)
