# utils.py
import math
import os
import random
import warnings
import time
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)
EPS = 1e-8

# --- Seed Setting ---
def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Potentially slow down training, but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

# --- Dynamic Path Generation ---
def get_dynamic_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Constructs paths dynamically based on config settings."""
    paths = {}
    base_ckpt_dir = config.get('paths', {}).get('checkpoints', './checkpoints')
    base_log_dir = config.get('paths', {}).get('logs', './logs')
    base_semantic_dir = config.get('paths', {}).get('semantic_features_dir', './semantic_features')

    # --- VLM related ---
    variant = config.get('model', {}).get('foundation_model', {}).get('variant', 'UNKNOWN_VLM')
    variant_safe = variant.replace('/', '-') # For filenames/dirs
    text_type = config.get('semantics', {}).get('generation', {}).get('text_type', 'unknown_text')
    llm_name = config.get('semantics', {}).get('generation', {}).get('llm', 'unknownLLM')
    llm_name_safe = llm_name.lower().replace('.', '').replace('-', '') # Sanitize LLM name

    # VLM weights path
    paths['vlm_weights'] = os.path.join(base_ckpt_dir, 'foundation_models', f"RemoteCLIP-{variant}.pt")

    # Semantic features path
    semantic_filename = f"hrrp_semantics_{variant_safe}_{text_type}"
    if text_type == 'llm_generated':
         semantic_filename += f"_{llm_name_safe}"
    semantic_filename += ".pth"
    paths['semantic_features'] = os.path.join(base_semantic_dir, semantic_filename)

    # --- CMSA-HRRP (Adapter Version) Paths ---
    # Experiment name based on VLM and semantics used for training adapter
    adapter_exp_name = f"{variant_safe}_{text_type}"
    if text_type == 'llm_generated': adapter_exp_name += f"_{llm_name_safe}"

    paths['adapter_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'hrrp_adapter', adapter_exp_name)
    paths['adapter_best_ckpt'] = os.path.join(paths['adapter_checkpoint_dir'], 'best.pth')
    paths['adapter_latest_ckpt'] = os.path.join(paths['adapter_checkpoint_dir'], 'latest.pth')

    paths['base_centers_path'] = os.path.join(base_ckpt_dir, f'base_centers_mean_{adapter_exp_name}.pth') # Centers depend on adapter+VLM

    paths['semalign_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'semalign_module', adapter_exp_name) # SemAlign trained based on adapter+VLM
    paths['semalign_best_ckpt'] = os.path.join(paths['semalign_checkpoint_dir'], 'best.pth')
    paths['semalign_latest_ckpt'] = os.path.join(paths['semalign_checkpoint_dir'], 'latest.pth')

    # Log paths
    paths['adapter_log_dir'] = os.path.join(base_log_dir, 'adapter_training', adapter_exp_name)
    paths['semalign_log_dir'] = os.path.join(base_log_dir, 'semalign_training_stage3', adapter_exp_name)
    paths['centers_log_dir'] = os.path.join(base_log_dir, 'compute_centers', adapter_exp_name)
    paths['fsl_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_adapter', adapter_exp_name) # Base dir for test logs

    # --- Baseline Paths ---
    baseline_exp_name = config.get('baseline_experiment_name', 'default_cnn')

    paths['baseline_cnn_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'hrrp_encoder_baseline', baseline_exp_name)
    paths['baseline_cnn_best_ckpt'] = os.path.join(paths['baseline_cnn_checkpoint_dir'], 'best.pth')
    paths['baseline_cnn_latest_ckpt'] = os.path.join(paths['baseline_cnn_checkpoint_dir'], 'latest.pth')

    paths['baseline_centers_1dcnn_path'] = os.path.join(base_ckpt_dir, f'base_centers_1dcnn_{baseline_exp_name}.pth')

    # Fusion module for baseline depends on baseline CNN and the semantics it's trained with
    fusion_1dcnn_exp_name = f"{baseline_exp_name}_{text_type}"
    if text_type == 'llm_generated': fusion_1dcnn_exp_name += f"_{llm_name_safe}"

    paths['baseline_fusion_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'fusion_module_1dcnn', fusion_1dcnn_exp_name)
    paths['baseline_fusion_best_ckpt'] = os.path.join(paths['baseline_fusion_checkpoint_dir'], 'best.pth')
    paths['baseline_fusion_latest_ckpt'] = os.path.join(paths['baseline_fusion_checkpoint_dir'], 'latest.pth')

    # Baseline Log paths
    paths['baseline_cnn_log_dir'] = os.path.join(base_log_dir, 'baseline_cnn_training', baseline_exp_name)
    paths['baseline_centers_1dcnn_log_dir'] = os.path.join(base_log_dir, 'compute_centers_1dcnn', baseline_exp_name)
    paths['baseline_fusion_log_dir'] = os.path.join(base_log_dir, 'fusion_1dcnn_training', fusion_1dcnn_exp_name)
    paths['protonet_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_protonet', baseline_exp_name)
    paths['1dcnn_semantics_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_1dcnn_semantics', fusion_1dcnn_exp_name)

    # Create directories if they don't exist (optional, scripts can also do this)
    # for path_key, path_val in paths.items():
    #     if 'dir' in path_key:
    #         os.makedirs(path_val, exist_ok=True)

    return paths


# --- Classifiers ---
def proto_classifier(prototypes: torch.Tensor, query_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classifies query features based on Euclidean distance to prototypes."""
    # prototypes: (N_way, FeatureDim)
    # query_x: (N_query, FeatureDim)
    query_x_expanded = query_x.unsqueeze(1) # (N_query, 1, FeatureDim)
    prototypes_expanded = prototypes.unsqueeze(0) # (1, N_way, FeatureDim)
    # Calculate negative squared Euclidean distance
    dist = -torch.sum((query_x_expanded - prototypes_expanded).pow(2), dim=2) # (N_query, N_way)
    predict = torch.argmax(dist, dim=1)
    return dist, predict

def Cosine_classifier(prototypes: torch.Tensor, query_x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classifies query features based on cosine similarity to prototypes."""
    # prototypes: (N_way, FeatureDim)
    # query_x: (N_query, FeatureDim)
    # Normalize features and prototypes INTERNALLY
    proto_normalized = normalize(prototypes) # (N_way, FeatureDim)
    query_normalized = normalize(query_x) # (N_query, FeatureDim)
    # Calculate cosine similarity (dot product of normalized vectors)
    logits = torch.mm(query_normalized, proto_normalized.t()) / temperature # (N_query, N_way)
    predict = torch.argmax(logits, dim=1)
    return logits, predict

def LR(support: torch.Tensor, support_y: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Logistic Regression classifier (requires scikit-learn)."""
    try:
        support_np = support.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_np = query.detach().cpu().numpy()

        clf = LogisticRegression(penalty='l2',
                                 random_state=0,
                                 C=1.0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        clf.fit(support_np, support_y_np)
        predict = clf.predict(query_np)
        return torch.tensor(predict)
    except Exception as e:
        logger.error(f"Logistic Regression failed: {e}")
        return torch.zeros(query.size(0), dtype=torch.long)


# --- Normalization ---
def normalize(x: torch.Tensor, epsilon: float = EPS) -> torch.Tensor:
    """L2 normalizes a tensor along the last dimension."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + epsilon)


# --- Contrastive Loss ---
def calculate_infonce_loss(features: torch.Tensor,
                           labels: torch.Tensor,
                           temperature: float = 0.1,
                           epsilon: float = EPS) -> torch.Tensor:
    """
    Calculates the supervised instance-based InfoNCE loss within a batch.

    Args:
        features (torch.Tensor): Normalized features (B, D).
        labels (torch.Tensor): Ground truth labels (B).
        temperature (float): Temperature scaling factor.
        epsilon (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    device = features.device
    batch_size = features.shape[0]

    # Create masks for positive and negative pairs
    labels = labels.contiguous().view(-1, 1)
    # Mask for positive pairs (same label, different instance)
    mask_pos = torch.eq(labels, labels.T).float().to(device)
    # Mask for diagonal elements (self-comparison)
    mask_diag = torch.eye(batch_size, dtype=torch.float32, device=device)
    # Positive mask excludes self-comparison
    mask_pos = mask_pos - mask_diag

    # Mask for negative pairs (different label)
    mask_neg = 1.0 - mask_pos - mask_diag

    # Calculate cosine similarity matrix
    # features are assumed to be normalized already
    sim_matrix = torch.matmul(features, features.T) # (B, B)

    # Apply temperature scaling
    sim_matrix = sim_matrix / temperature

    # --- Numerator: Sum of positive similarities ---
    # For each anchor, sum similarities with its positive pairs
    # Use exp(sim) for the sum
    exp_sim = torch.exp(sim_matrix)
    pos_sum = torch.sum(exp_sim * mask_pos, dim=1) # (B,)

    # --- Denominator: Sum of positive and negative similarities ---
    # For each anchor, sum similarities with all *other* samples (pos + neg)
    # We exclude self-similarity from the denominator sum
    all_other_mask = 1.0 - mask_diag
    all_other_sum = torch.sum(exp_sim * all_other_mask, dim=1) # (B,)

    # --- Calculate Loss ---
    # Loss for each anchor: -log(pos_sum / all_other_sum)
    # Handle cases where pos_sum might be zero (no other positive samples in batch)
    # Add epsilon to denominator for stability
    loss_per_anchor = -torch.log(pos_sum / (all_other_sum + epsilon) + epsilon)

    # Average loss over anchors that HAVE positive pairs
    # Count how many positive pairs each anchor has
    num_pos_pairs = torch.sum(mask_pos, dim=1) # (B,)
    # Only compute loss for anchors with at least one positive pair
    valid_anchors_mask = (num_pos_pairs > 0).float()
    # Average loss only over valid anchors
    loss = torch.sum(loss_per_anchor * valid_anchors_mask) / (torch.sum(valid_anchors_mask) + epsilon)

    # Alternative: Average over all anchors, assigning 0 loss if no pos pairs?
    # loss = loss_per_anchor.mean() # Simpler, but might dilute signal if many anchors lack pos pairs

    return loss


# --- Averaging and Timing ---
class Averager():
    """Computes the running average of values."""
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Timer():
    """Measures execution time."""
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


# --- Accuracy Calculation ---
def count_95acc(accuracies: np.ndarray) -> Tuple[float, float]:
    """Calculates mean accuracy and 95% confidence interval."""
    if accuracies is None or len(accuracies) == 0:
        return 0.0, 0.0
    acc_avg = np.mean(accuracies)
    acc_ci95 = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
    return acc_avg, acc_ci95

# --- Config Loading Helper ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            raise exc
    return config

# --- Truncated Normal (Keep as is) ---
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
