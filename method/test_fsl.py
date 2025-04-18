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
from method.alignment import SemAlignModule # Use the renamed module
from logger import loggers
from utils import set_seed, Cosine_classifier, count_95acc, normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def main(config, args_override):
    """Main Few-Shot Learning evaluation function using Adapter + SemAlign."""

    # --- Setup ---
    n_way = args_override.n_way if args_override.n_way is not None else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot is not None else config['fsl']['k_shot']
    q_query = args_override.q_query if args_override.q_query is not None else config['fsl']['q_query']
    test_episodes = args_override.test_episodes if args_override.test_episodes is not None else config['fsl']['test_episodes']
    kappa = args_override.kappa if args_override.kappa is not None else config['model']['fusion_module']['kappa'] # Use kappa from fusion section
    # --- Checkpoint Paths ---
    adapter_checkpoint_path = args_override.adapter_checkpoint if args_override.adapter_checkpoint else os.path.join(config['paths']['checkpoints'], 'hrrp_adapter', 'latest.pth')
    semalign_checkpoint_path = args_override.semalign_checkpoint if args_override.semalign_checkpoint else os.path.join(config['paths']['checkpoints'], 'semalign_module', 'latest.pth')

    # Setup Log directory
    setting_name = f"AdapterSemAlign_{n_way}way_{k_shot}shot_k{kappa:.2f}"
    log_dir = os.path.join(config['paths'].get('logs', './logs'), 'fsl_testing_adapter_semalign', setting_name) # New log subdir
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'test_fsl_adapter_semalign.log')
    loggers(log_file_path)
    logger.info(f"Logger configured. Logging to: {log_file_path}")

    logger.info(f"Loaded base configuration: {config}")
    logger.info(f"Testing with Overrides: N-Way={n_way}, K-Shot={k_shot}, Q-Query={q_query}, Episodes={test_episodes}, Kappa={kappa}, AdapterCkpt={adapter_checkpoint_path}, SemAlignCkpt={semalign_checkpoint_path}")

    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading (Novel Set) ---
    logger.info("Loading novel dataset...")
    try:
        novel_dataset = HRRPDataset(
            root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
            split='novel', classes=config['data']['novel_classes'], target_length=config['data']['target_length'],
            normalization=config['data']['normalization'], phase_info='magnitude'
        )
        if len(novel_dataset.target_classes) < n_way: sys.exit(f"Not enough novel classes for {n_way}-way.")
        min_samples = min(len([i for i, l in enumerate(novel_dataset.labels) if novel_dataset.idx_to_class[l] == cn]) for cn in novel_dataset.target_classes)
        if min_samples < k_shot + q_query: sys.exit(f"Not enough samples per class. Required={k_shot + q_query}, Min={min_samples}.")
    except Exception as e: logger.error(f"Failed to load Novel HRRPDataset: {e}"); sys.exit(1)

    novel_sampler = CategoriesSampler(novel_dataset.labels, test_episodes, n_way, k_shot + q_query)
    novel_loader = DataLoader(novel_dataset, batch_sampler=novel_sampler, num_workers=config.get('num_workers', 0), pin_memory=True)
    logger.info(f"Novel dataset loaded: {len(novel_dataset)} samples, {len(novel_dataset.target_classes)} classes.")

    # --- Load Semantic Features (Novel Classes) ---
    logger.info(f"Loading semantic features from: {config['semantics']['feature_path']}")
    try:
        semantic_data = torch.load(config['semantics']['feature_path'], map_location='cpu')
        semantic_features = {k: v.float().to(device) for k, v in semantic_data['semantic_feature'].items() if k in config['data']['novel_classes']}
        logger.info(f"Loaded semantic features for {len(semantic_features)} novel classes.")
    except Exception as e: logger.error(f"Error loading semantic features: {e}"); sys.exit(1)

    # --- Load VLM (Visual + Text Encoders - Frozen) ---
    fm_config = config['model']['foundation_model']
    logger.info(f"Loading VLM: {fm_config['name']} ({fm_config['variant']})")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            fm_variant = fm_config['variant']
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"
            local_weights_path = os.path.join(base_checkpoint_dir, 'foundation_models', weights_filename)
            if not os.path.exists(local_weights_path): logger.error(f"VLM weights not found: {local_weights_path}"); sys.exit(1)
            logger.info(f"Loading VLM weights from: {local_weights_path}")

            vlm_model, _, vlm_preprocess = open_clip.create_model_and_transforms(fm_variant, pretrained=local_weights_path)
            visual_encoder = vlm_model.visual.to(device).eval() # f_V
            text_encoder = vlm_model.to(device).eval() # Keep full model for text encoding if needed, or just text part
            for param in vlm_model.parameters(): param.requires_grad = False
            logger.info("VLM loaded and frozen.")
            expected_img_size = visual_encoder.image_size if hasattr(visual_encoder, 'image_size') else \
                               (visual_encoder.img_size if hasattr(visual_encoder, 'img_size') else 224)
            if isinstance(expected_img_size, (tuple, list)): expected_img_size = expected_img_size[0]
            visual_dim = fm_config['text_encoder_dim'] # Assuming CLIP symmetry
            semantic_dim = fm_config['text_encoder_dim']
        else: logger.error(f"Unsupported VLM: {fm_config['name']}"); sys.exit(1)
    except Exception as e: logger.error(f"Failed to load VLM: {e}"); sys.exit(1)

    # --- Load Trained Adapter (Frozen) ---
    logger.info(f"Loading trained adapter from: {adapter_checkpoint_path}")
    try:
        adapter = HRPPtoPseudoImage(
            hrrp_length=config['data']['target_length'], input_channels=1, output_channels=3, output_size=expected_img_size
        ).to(device)
        adapter_checkpoint = torch.load(adapter_checkpoint_path, map_location=device)
        adapter.load_state_dict(adapter_checkpoint['adapter_state_dict'])
        adapter.eval(); [p.requires_grad_(False) for p in adapter.parameters()]
        logger.info("Adapter loaded and frozen.")
    except Exception as e: logger.error(f"Error loading adapter: {e}"); sys.exit(1)

    # --- Load Trained SemAlign Module (Frozen) ---
    logger.info(f"Loading trained SemAlign module from: {semalign_checkpoint_path}")
    try:
        semalign_module = SemAlignModule(
            visual_dim=visual_dim, semantic_dim=semantic_dim,
            hidden_dim=config['model']['fusion_module']['hidden_dim'], output_dim=visual_dim
        ).to(device)
        semalign_checkpoint = torch.load(semalign_checkpoint_path, map_location=device)
        semalign_module.load_state_dict(semalign_checkpoint['semalign_state_dict'])
        semalign_module.eval(); [p.requires_grad_(False) for p in semalign_module.parameters()]
        logger.info("SemAlign module loaded and frozen.")
    except Exception as e: logger.error(f"Error loading SemAlign module: {e}"); sys.exit(1)

    # --- Evaluation Loop ---
    logger.info("Starting few-shot evaluation...")
    all_accuracies = []
    query_labels_proto = torch.arange(n_way).repeat_interleave(q_query).long().to(device)

    pbar = tqdm(novel_loader, total=test_episodes, desc="Evaluating Episodes")
    for episode_idx, batch_item in enumerate(pbar):
        if episode_idx >= test_episodes: break

        try:
            # Data loading (yields data_batch, label_batch)
            hrrp_data_all, labels_all_original_tensor = batch_item
            hrrp_data_all = hrrp_data_all.to(device)
            labels_all_original = labels_all_original_tensor.tolist()

            # Map labels
            support_original_labels = labels_all_original[:n_way * k_shot]
            unique_labels_in_episode = sorted(list(set(support_original_labels)))
            if len(unique_labels_in_episode) < n_way: logger.warning(f"Ep {episode_idx}: Not enough classes. Skip."); continue
            label_map = {orig_label: episode_label for episode_label, orig_label in enumerate(unique_labels_in_episode)}
            episode_labels_all = torch.tensor([label_map[l] for l in labels_all_original]).to(device)

        except Exception as e: logger.error(f"Error processing batch data ep {episode_idx}: {e}"); continue

        # --- Feature Extraction ---
        with torch.no_grad():
            try:
                pseudo_images_all = adapter(hrrp_data_all)
                if pseudo_images_all.shape[-2:] != (expected_img_size, expected_img_size):
                     pseudo_images_all = F.interpolate(pseudo_images_all, size=(expected_img_size, expected_img_size), mode='bilinear', align_corners=False)
                visual_features_all = visual_encoder(pseudo_images_all) # z_V
                if isinstance(visual_features_all, tuple): visual_features_all = visual_features_all[0]
                if visual_features_all.dim() == 3 and visual_features_all.shape[1] > 1: visual_features_all = visual_features_all[:, 0]
                # NO normalization here for z_V input to SemAlign/mean calculation, matching SemFew paper
                if visual_features_all.dim() != 2: logger.error(f"Visual features shape error: {visual_features_all.shape}. Skip."); continue
            except Exception as e: logger.error(f"Error in feature extraction ep {episode_idx}: {e}"); continue

        # Split support and query
        support_features = visual_features_all[:n_way * k_shot] # z_V_sup
        query_features = visual_features_all[n_way * k_shot:]   # z_V_query
        support_labels_episode = episode_labels_all[:n_way * k_shot]

        # --- Calculate Prototypes ---
        prototypes = []
        prototype_calculation_failed = False
        with torch.no_grad():
            for c in range(n_way):
                class_mask = (support_labels_episode == c)
                if not torch.any(class_mask): logger.warning(f"Ep {episode_idx}: No support for class {c}. Skip."); prototype_calculation_failed = True; break
                class_support_features = support_features[class_mask] # z_V for class c

                original_label = unique_labels_in_episode[c]
                class_name = novel_dataset.idx_to_class[original_label]

                # Mean visual feature (ut in SemFew Eq. 6)
                mean_visual_feature = class_support_features.mean(dim=0).squeeze() # (visual_dim,)
                if mean_visual_feature.dim() != 1: logger.error(f"Ep {episode_idx}, Cls {c}: Mean visual not 1D. Skip."); prototype_calculation_failed = True; break

                # Reconstructed prototype (rt in SemFew Eq. 5 & 6)
                if class_name in semantic_features:
                    class_semantic_feature = semantic_features[class_name].squeeze() # (semantic_dim,)
                    if class_semantic_feature.dim() != 1:
                         logger.error(f"Ep {episode_idx}, Cls {c}: Semantic not 1D. Using mean only.");
                         reconstructed_prototype = mean_visual_feature # Fallback
                    else:
                         repeated_semantics = class_semantic_feature.unsqueeze(0).repeat(class_support_features.size(0), 1)
                         # Use SemAlign module 'h'
                         reconstructed_features_per_sample = semalign_module(class_support_features, repeated_semantics) # h([z_V_sup_i, z_T_sup_c]) -> (K, visual_dim)
                         reconstructed_prototype = reconstructed_features_per_sample.mean(dim=0).squeeze() # rt = mean(h(...)) -> (visual_dim,)
                         if reconstructed_prototype.dim() != 1:
                             logger.error(f"Ep {episode_idx}, Cls {c}: Reconstructed proto not 1D. Using mean only.");
                             reconstructed_prototype = mean_visual_feature # Fallback
                else:
                    logger.warning(f"Ep {episode_idx}: Semantic for '{class_name}' missing. Using mean only for rt.")
                    reconstructed_prototype = mean_visual_feature # Use mean as fallback if semantics missing

                # Final prototype (pt in SemFew Eq. 6)
                final_prototype = float(kappa) * reconstructed_prototype + (1.0 - float(kappa)) * mean_visual_feature # pt = k*rt + (1-k)*ut

                if final_prototype.dim() != 1: logger.warning(f"Ep {episode_idx}, Cls {c}: Final proto not 1D. Squeezing."); final_prototype = final_prototype.squeeze()
                if final_prototype.dim() != 1: logger.error(f"Ep {episode_idx}, Cls {c}: Cannot make final proto 1D. Skip."); prototype_calculation_failed = True; break

                prototypes.append(final_prototype)

        if prototype_calculation_failed: continue

        # --- Stack Prototypes and Classify ---
        try:
            prototypes = torch.stack(prototypes) # Shape: (n_way, visual_dim)
            if prototypes.dim() != 2 or prototypes.size(0) != n_way or prototypes.size(1) != visual_dim:
                 logger.error(f"Stacked protos shape error: {prototypes.shape}. Expected ({n_way}, {visual_dim}). Skip."); continue
        except Exception as e: logger.error(f"Error stacking protos: {e}. Shapes: {[p.shape for p in prototypes]}. Skip."); continue

        # Classify Query Set (using VISUAL features z_V_query against pt)
        with torch.no_grad():
            # Note: Query features z_V_query were NOT normalized earlier if inputting to h
            # But prototypes pt are combinations of z_V and h output.
            # For Cosine_classifier, both inputs should be normalized.
            query_features_norm = normalize(query_features)
            prototypes_norm = normalize(prototypes)

            if query_features_norm.shape[-1] != prototypes_norm.shape[-1]:
                 logger.error(f"Query dim ({query_features_norm.shape[-1]}) != proto dim ({prototypes_norm.shape[-1]}). Skip."); continue

            logits, predictions = Cosine_classifier(prototypes_norm, query_features_norm)
            accuracy = (predictions == query_labels_proto).float().mean().item()
            all_accuracies.append(accuracy)

        pbar.set_postfix({'Acc': f"{accuracy*100:.2f}%"})

    # --- Final Results ---
    if not all_accuracies: logger.error("No episodes evaluated."); return
    mean_acc, conf_interval = count_95acc(np.array(all_accuracies))
    logger.info(f"--- Final Results ({setting_name}) ---")
    logger.info(f"Evaluated {len(all_accuracies)} episodes.")
    logger.info(f"Average Accuracy: {mean_acc * 100:.2f}%")
    logger.info(f"95% Confidence Interval: {conf_interval * 100:.2f}%")
    logger.info(f"Result: {mean_acc * 100:.2f} ± {conf_interval * 100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot HRRP Recognition (Adapter + SemAlign)")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the configuration file')
    parser.add_argument('--adapter_checkpoint', type=str, default=None, help='Path to trained ADAPTER checkpoint (overrides config default)')
    parser.add_argument('--semalign_checkpoint', type=str, default=None, help='Path to trained SEMALIGN checkpoint (overrides config default)')
    parser.add_argument('--n_way', type=int, default=None, help='N-way (overrides config)')
    parser.add_argument('--k_shot', type=int, default=None, help='K-shot (overrides config)')
    parser.add_argument('--q_query', type=int, default=None, help='Query samples per class (overrides config)')
    parser.add_argument('--test_episodes', type=int, default=None, help='Number of test episodes (overrides config)')
    parser.add_argument('--kappa', type=float, default=None, help='Prototype fusion factor kappa (overrides config)')
    # --no_align is removed as it's not applicable here

    args = parser.parse_args()
    if not os.path.exists(args.config): logging.error(f"Config not found: {args.config}"); sys.exit(1)
    configuration = load_config(args.config)
    main(configuration, args)