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
from model.hrrp_encoder import HRRPEncoder # Assuming this exists
from method.alignment import AlignmentModule, FusionModule
from logger import loggers
from utils import set_seed, Cosine_classifier, count_95acc, count_kacc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config, args_override):
    """Main Few-Shot Learning evaluation function."""

    # --- Setup ---
    # Use command-line args for FSL settings, overriding config if provided
    n_way = args_override.n_way if args_override.n_way else config['fsl']['n_way']
    k_shot = args_override.k_shot if args_override.k_shot else config['fsl']['k_shot']
    q_query = args_override.q_query if args_override.q_query else config['fsl']['q_query']
    test_episodes = args_override.test_episodes if args_override.test_episodes else config['fsl']['test_episodes']
    kappa = args_override.kappa if args_override.kappa is not None else config['model']['fusion_module']['kappa']
    checkpoint_path = args_override.checkpoint if args_override.checkpoint else os.path.join(config['paths']['checkpoints'], 'alignment_module', 'latest.pth')

    # Determine if alignment is used based on flag
    use_alignment = not args_override.no_align

    # Setup Log directory based on test settings
    setting_name = f"{n_way}way_{k_shot}shot_k{kappa:.2f}"
    if not use_alignment:
        setting_name += "_NoAlign"
    log_dir = os.path.join(config['paths']['logs'], 'fsl_testing', setting_name)
    os.makedirs(log_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, 'test_fsl'))
    log.info(f"Loaded base configuration: {config}")
    log.info(f"Testing with Overrides: N-Way={n_way}, K-Shot={k_shot}, Q-Query={q_query}, Episodes={test_episodes}, Kappa={kappa}, UseAlign={use_alignment}, Ckpt={checkpoint_path}")

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Data Loading (Novel Set) ---
    log.info("Loading novel dataset...")
    novel_dataset = HRRPDataset(
        root_dirs=[config['data']['simulated_path'], config['data']['measured_path']],
        split='novel',
        classes=config['data']['novel_classes'],
        target_length=config['data']['target_length'],
        normalization=config['data']['normalization'],
        phase_info='magnitude' # Match training
    )
    if len(novel_dataset.target_classes) < n_way:
         log.error(f"Not enough novel classes ({len(novel_dataset.target_classes)}) for {n_way}-way setting.")
         sys.exit(1)
    if len(novel_dataset) < n_way * (k_shot + q_query):
         log.warning(f"Novel dataset size ({len(novel_dataset)}) might be small for sampling {n_way}-way {k_shot+q_query}-samples per class.")

    novel_sampler = CategoriesSampler(
        novel_dataset.labels, # Use labels directly
        n_batch=test_episodes,
        n_cls=n_way,
        n_per=k_shot + q_query
    )
    novel_loader = DataLoader(
        novel_dataset,
        batch_sampler=novel_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    log.info(f"Novel dataset loaded with {len(novel_dataset)} samples across {len(novel_dataset.target_classes)} classes.")
    log.info(f"Sampling {test_episodes} episodes: {n_way}-way, {k_shot}-shot, {q_query}-query.")

    # --- Load Semantic Features (Novel Classes) ---
    log.info(f"Loading semantic features from: {config['semantics']['feature_path']}")
    try:
        semantic_data = torch.load(config['semantics']['feature_path'], map_location='cpu')
        # Filter for novel classes only
        semantic_features = {k: v.float().to(device)
                             for k, v in semantic_data['semantic_feature'].items()
                             if k in config['data']['novel_classes']}
        if len(semantic_features) != len(config['data']['novel_classes']):
             log.warning("Mismatch between loaded semantic features and configured novel classes.")
        log.info(f"Loaded semantic features for {len(semantic_features)} novel classes.")
    except FileNotFoundError:
        log.error(f"Semantic features file not found at {config['semantics']['feature_path']}")
        sys.exit(1)
    except KeyError:
         log.error(f"Key 'semantic_feature' not found in {config['semantics']['feature_path']}")
         sys.exit(1)

    # --- Model Initialization and Loading Weights ---
    hrrp_encoder = HRRPEncoder(output_dim=config['model']['hrrp_encoder']['output_dim']).to(device)
    alignment_module = AlignmentModule(
        hrrp_feat_dim=config['model']['hrrp_encoder']['output_dim'],
        semantic_dim=config['model']['foundation_model']['text_encoder_dim'],
        hidden_dim=config['model']['alignment_module']['hidden_dim']
    ).to(device)
    fusion_module = FusionModule(
        aligned_hrrp_dim=config['model']['foundation_model']['text_encoder_dim'],
        semantic_dim=config['model']['foundation_model']['text_encoder_dim'],
        hidden_dim=config['model']['fusion_module']['hidden_dim'],
        output_dim=config['model']['foundation_model']['text_encoder_dim'] # Match target space
    ).to(device)

    # --- Placeholder for NoAlign scenario ---
    # If not using alignment, potentially need a simple linear layer to match dimensions
    # This layer would ideally be trained similarly to h_A but maybe with a different target or loss
    # For simplicity in this baseline, we might just use raw features if dimensions match, or skip fusion.
    linear_proj_noalign = None
    if not use_alignment:
        hrrp_dim = config['model']['hrrp_encoder']['output_dim']
        semantic_dim = config['model']['foundation_model']['text_encoder_dim']
        if hrrp_dim != semantic_dim and kappa > 0: # Only needed if fusing and dims mismatch
            log.warning(f"[NoAlign] HRRP dim ({hrrp_dim}) != Semantic dim ({semantic_dim}). Adding Linear projection.")
            linear_proj_noalign = nn.Linear(hrrp_dim, semantic_dim).to(device)
            # Note: This linear layer is UNTRAINED in this simple baseline setup.
            # A proper NoAlign baseline might require training this projection layer.
        elif hrrp_dim != semantic_dim and kappa == 0:
             log.warning(f"[NoAlign, kappa=0] HRRP dim ({hrrp_dim}) != Semantic dim ({semantic_dim}). Cannot directly compare prototypes.")
             # This scenario likely needs adjustment - maybe classify in hrrp_dim space?
             # For now, we proceed assuming kappa=0 means classify using mean z_H.

    log.info(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hrrp_encoder.load_state_dict(checkpoint['hrrp_encoder_state_dict'])
        if use_alignment:
            alignment_module.load_state_dict(checkpoint['alignment_module_state_dict'])
        if kappa > 0: # Only load fusion module if needed
            fusion_module.load_state_dict(checkpoint['fusion_module_state_dict'])
        log.info(f"Weights loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    except FileNotFoundError:
        log.error(f"Checkpoint file not found at {checkpoint_path}. Cannot run evaluation.")
        sys.exit(1)
    except KeyError as e:
        log.error(f"Missing key in checkpoint file {checkpoint_path}: {e}. Check training script.")
        sys.exit(1)

    hrrp_encoder.eval()
    alignment_module.eval()
    fusion_module.eval()
    if linear_proj_noalign:
        linear_proj_noalign.eval()


    # --- Evaluation Loop ---
    log.info("Starting few-shot evaluation...")
    all_accuracies = []
    # Generate labels for query set within an episode
    query_labels_proto = torch.arange(n_way).repeat_interleave(q_query).long().to(device)

    pbar = tqdm(range(test_episodes), desc="Evaluating Episodes")
    for episode_idx in pbar:
        # DataLoader provides indices for one episode
        try:
            support_indices, query_indices = novel_loader.__iter__().__next__() # Get one batch of indices
        except StopIteration:
            log.warning("DataLoader exhausted before reaching target episodes.")
            break
        except Exception as e:
            log.error(f"Error getting batch from DataLoader: {e}")
            continue

        # Get actual data using indices
        try:
            all_indices = torch.cat([support_indices, query_indices])
            hrrp_data_all = []
            labels_all = []
            for idx in all_indices:
                sample, label = novel_dataset[idx.item()]
                hrrp_data_all.append(sample)
                labels_all.append(label) # Store original labels to map to episode labels later

            hrrp_data_all = torch.stack(hrrp_data_all).to(device)
            # Map original labels to 0..N-1 within the episode
            unique_labels_in_episode = sorted(list(set(labels_all[:n_way*k_shot]))) # Get the N classes in the support set
            label_map = {orig_label: episode_label for episode_label, orig_label in enumerate(unique_labels_in_episode)}
            episode_labels_all = torch.tensor([label_map[l] for l in labels_all]).to(device)

        except Exception as e:
            log.error(f"Error processing batch data for episode {episode_idx}: {e}")
            continue


        # Extract and Align Features
        with torch.no_grad():
            hrrp_features_all = hrrp_encoder(hrrp_data_all) # z_H

            if use_alignment:
                aligned_features_all = alignment_module(hrrp_features_all) # z'_H
            elif linear_proj_noalign: # NoAlign case with projection
                 aligned_features_all = linear_proj_noalign(hrrp_features_all) # Use projection output
            else: # NoAlign case, use raw features (potentially mismatching dims if kappa>0)
                 aligned_features_all = hrrp_features_all


        # Split support and query
        support_features = aligned_features_all[:n_way * k_shot]
        query_features = aligned_features_all[n_way * k_shot:]
        support_labels = episode_labels_all[:n_way * k_shot]
        # query_labels should match query_labels_proto

        # Calculate Prototypes
        prototypes = []
        with torch.no_grad():
            for c in range(n_way):
                # Features for current class
                class_support_features = support_features[support_labels == c]
                # Original class name corresponding to episode class 'c'
                original_label = unique_labels_in_episode[c]
                class_name = novel_dataset.idx_to_class[original_label]

                # Mean aligned feature (u_c)
                mean_aligned_feature = class_support_features.mean(dim=0)

                if kappa > 0 and class_name in semantic_features:
                    # Semantic Enhancement
                    class_semantic_feature = semantic_features[class_name].squeeze(0) # Ensure (dim,)

                    # Repeat semantic feature for each support sample of the class
                    repeated_semantics = class_semantic_feature.unsqueeze(0).repeat(class_support_features.size(0), 1)

                    # Compute reconstructed features per sample
                    # Note: FusionModule expects (Batch, Dim)
                    reconstructed = fusion_module(class_support_features, repeated_semantics)

                    # Mean reconstructed prototype (r_c)
                    mean_reconstructed_prototype = reconstructed.mean(dim=0)

                    # Final prototype (p_c)
                    final_prototype = kappa * mean_reconstructed_prototype + (1 - kappa) * mean_aligned_feature
                else:
                    # Use only mean aligned feature (kappa=0 or semantic feature missing)
                    if kappa > 0 and class_name not in semantic_features:
                         log.warning(f"Semantic feature for class '{class_name}' not found. Using kappa=0 for this class.")
                    final_prototype = mean_aligned_feature

                prototypes.append(final_prototype)

            prototypes = torch.stack(prototypes) # Shape: (n_way, feature_dim)

        # Classify Query Set
        with torch.no_grad():
            logits, predictions = Cosine_classifier(prototypes, query_features)
            accuracy = (predictions == query_labels_proto).float().mean().item()
            all_accuracies.append(accuracy)

        pbar.set_postfix({'Acc': f"{accuracy*100:.2f}%"})


    # --- Final Results ---
    if not all_accuracies:
         log.error("No episodes were successfully evaluated.")
         return

    mean_acc, conf_interval = count_95acc(np.array(all_accuracies))
    log.info(f"--- Final Results ({setting_name}) ---")
    log.info(f"Average Accuracy: {mean_acc * 100:.2f}%")
    log.info(f"95% Confidence Interval: {conf_interval * 100:.2f}%")
    log.info(f"Result: {mean_acc * 100:.2f} ± {conf_interval * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot HRRP Recognition")
    parser.add_argument('--config', type=str, default='hrrp_fsl_config.yaml', help='Path to the configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the trained model checkpoint (overrides config)')
    parser.add_argument('--n_way', type=int, default=None, help='N-way (overrides config)')
    parser.add_argument('--k_shot', type=int, default=None, help='K-shot (overrides config)')
    parser.add_argument('--q_query', type=int, default=None, help='Query samples per class (overrides config)')
    parser.add_argument('--test_episodes', type=int, default=None, help='Number of test episodes (overrides config)')
    parser.add_argument('--kappa', type=float, default=None, help='Prototype fusion factor kappa (overrides config)')
    parser.add_argument('--no_align', action='store_true', help='Run the NoAlign baseline (bypass alignment module)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    main(configuration, args)