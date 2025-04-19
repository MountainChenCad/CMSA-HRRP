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
import yaml # Added for loading config within utils if needed (or pass config dict)

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
    base_ckpt_dir = config['paths']['checkpoints']
    base_log_dir = config['paths']['logs']
    base_semantic_dir = config['paths']['semantic_features_dir']

    # --- VLM related ---
    variant = config['model']['foundation_model']['variant']
    variant_safe = variant.replace('/', '-') # For filenames/dirs
    text_type = config['semantics']['generation']['text_type']
    llm_name = config['semantics']['generation'].get('llm', 'unknownLLM').lower().replace('.', '')

    # VLM weights path
    paths['vlm_weights'] = os.path.join(base_ckpt_dir, 'foundation_models', f"RemoteCLIP-{variant}.pt")

    # Semantic features path
    semantic_filename = f"hrrp_semantics_{variant_safe}_{text_type}.pth"
    if text_type == 'llm_generated': # Optionally add LLM name if generated
         semantic_filename = f"hrrp_semantics_{variant_safe}_{text_type}_{llm_name}.pth"
    paths['semantic_features'] = os.path.join(base_semantic_dir, semantic_filename)

    # --- CMSA-HRRP (Adapter Version) Paths ---
    adapter_exp_name = f"{variant_safe}_{text_type}" # Experiment name based on VLM and semantics used for training
    paths['adapter_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'hrrp_adapter', adapter_exp_name)
    paths['adapter_best_ckpt'] = os.path.join(paths['adapter_checkpoint_dir'], 'best.pth')
    paths['adapter_latest_ckpt'] = os.path.join(paths['adapter_checkpoint_dir'], 'latest.pth')

    paths['base_centers_path'] = os.path.join(base_ckpt_dir, f'base_centers_mean_{adapter_exp_name}.pth') # Centers depend on adapter+VLM used

    paths['semalign_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'semalign_module', adapter_exp_name) # SemAlign trained based on adapter+VLM
    paths['semalign_best_ckpt'] = os.path.join(paths['semalign_checkpoint_dir'], 'best.pth')
    paths['semalign_latest_ckpt'] = os.path.join(paths['semalign_checkpoint_dir'], 'latest.pth')

    # Log paths (example structure, adjust as needed)
    paths['adapter_log_dir'] = os.path.join(base_log_dir, 'adapter_training', adapter_exp_name)
    paths['semalign_log_dir'] = os.path.join(base_log_dir, 'semalign_training_stage3', adapter_exp_name)
    paths['centers_log_dir'] = os.path.join(base_log_dir, 'compute_centers', adapter_exp_name)
    paths['fsl_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_adapter', adapter_exp_name) # Base dir for test logs

    # --- Baseline Paths (Construct dynamically) ---
    # Baseline paths might be independent of VLM variant/text_type, or configurable
    baseline_exp_name = config.get('baseline_experiment_name', 'default_cnn')  # Add baseline name to config?

    paths['baseline_cnn_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'hrrp_encoder_baseline', baseline_exp_name)
    paths['baseline_cnn_best_ckpt'] = os.path.join(paths['baseline_cnn_checkpoint_dir'], 'best.pth')
    paths['baseline_cnn_latest_ckpt'] = os.path.join(paths['baseline_cnn_checkpoint_dir'], 'latest.pth')

    paths['baseline_centers_1dcnn_path'] = os.path.join(base_ckpt_dir, f'base_centers_1dcnn_{baseline_exp_name}.pth')

    # Fusion module for baseline depends on baseline CNN and semantics used
    # Let's tie its name to baseline cnn name and the text_type used for fusion
    fusion_1dcnn_exp_name = f"{baseline_exp_name}_{text_type}"
    paths['baseline_fusion_checkpoint_dir'] = os.path.join(base_ckpt_dir, 'fusion_module_1dcnn', fusion_1dcnn_exp_name)
    paths['baseline_fusion_best_ckpt'] = os.path.join(paths['baseline_fusion_checkpoint_dir'], 'best.pth')
    paths['baseline_fusion_latest_ckpt'] = os.path.join(paths['baseline_fusion_checkpoint_dir'], 'latest.pth')

    # Baseline Log paths
    paths['baseline_cnn_log_dir'] = os.path.join(base_log_dir, 'baseline_cnn_training', baseline_exp_name)
    paths['baseline_centers_1dcnn_log_dir'] = os.path.join(base_log_dir, 'compute_centers_1dcnn', baseline_exp_name)
    paths['baseline_fusion_log_dir'] = os.path.join(base_log_dir, 'fusion_1dcnn_training', fusion_1dcnn_exp_name)
    paths['protonet_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_protonet',
                                                  baseline_exp_name)  # Base dir for test logs
    paths['1dcnn_semantics_test_log_dir'] = os.path.join(base_log_dir, 'fsl_testing_1dcnn_semantics',
                                                         fusion_1dcnn_exp_name)  # Base dir for test logs

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
    # Normalize features and prototypes
    proto_normalized = F.normalize(prototypes, p=2, dim=-1) # (N_way, FeatureDim)
    query_normalized = F.normalize(query_x, p=2, dim=-1) # (N_query, FeatureDim)
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

# --- Config Loading Helper (Optional, scripts can load directly too) ---
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
# ... (copy the _no_grad_trunc_normal_ and trunc_normal_ functions here) ...
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
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
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    """Fills the input Tensor with values drawn from a truncated
    normal distribution.

    Args:
        tensor: an n-dimensional torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
