import math
import os
import random
import warnings
import time # Added for Timer class
import logging
from typing import Tuple, List, Dict, Any # Add Tuple here, List/Dict/Any might be useful too
import numpy as np
import torch
# from sklearn.cluster import KMeans # Removed dependency if cluster function is not used
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
# from torchvision import transforms # Removed dependency, HRRP uses custom transforms/preprocessing

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Potentially slow down training, but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


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
        # Return dummy predictions or raise error
        return torch.zeros(query.size(0), dtype=torch.long)


# --- Normalization ---
def normalize(x: torch.Tensor, epsilon: float = EPS) -> torch.Tensor:
    """L2 normalizes a tensor along the last dimension."""
    # x: (..., d)
    x = x / (x.norm(p=2, dim=-1, keepdim=True) + epsilon)
    return x


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

def count_kacc(proto: torch.Tensor, com_proto: torch.Tensor, query: torch.Tensor, k: float, query_labels: torch.Tensor, n_way: int, classifier=Cosine_classifier) -> float:
    """Calculates accuracy using a convex combination of two prototypes."""
    # proto: Original prototypes (e.g., mean aligned HRRP), shape (N_way, FeatureDim)
    # com_proto: Complementary prototypes (e.g., mean reconstructed), shape (N_way, FeatureDim)
    # query: Query features, shape (N_query, FeatureDim)
    # k: Fusion factor (0 to 1)
    # query_labels: Ground truth labels for query set, shape (N_query,) (values 0 to N_way-1)
    # n_way: Number of classes in the episode

    if proto.shape != com_proto.shape:
        raise ValueError("Prototype shapes must match for combination.")

    # Ensure k is a float
    k_float = float(k)

    # Calculate the fused prototype
    fused_proto = k_float * com_proto + (1 - k_float) * proto # Use com_proto for k portion as in SemFew paper logic

    # Classify using the fused prototype
    _, predict = classifier(fused_proto, query)

    # Calculate accuracy
    if predict.shape != query_labels.shape:
         raise ValueError(f"Prediction shape {predict.shape} does not match label shape {query_labels.shape}")

    accuracy = (predict == query_labels).float().mean().item()
    return accuracy


# --- Clustering (Optional - Kept from original, needs scikit-learn) ---
# def cluster(data: Dict[Any, List[np.ndarray]], n_clusters: int, num_samples_per_class: int) -> Dict[Any, np.ndarray]:
#     """Performs KMeans clustering to find representative centers (requires scikit-learn)."""
#     try:
#         from sklearn.cluster import KMeans
#     except ImportError:
#         logger.error("Scikit-learn not found. Skipping cluster function.")
#         return {}

#     x_all = []
#     original_labels = []
#     class_indices = []
#     current_index = 0

#     # Prepare data for KMeans: flatten features and track original labels/indices
#     for label, features in data.items():
#         if len(features) != num_samples_per_class:
#              logger.warning(f"Class {label} has {len(features)} samples, expected {num_samples_per_class}. Clustering might be skewed.")
#         x_all.extend(features)
#         original_labels.append(label)
#         class_indices.append((current_index, current_index + len(features)))
#         current_index += len(features)

#     if not x_all:
#         logger.error("No data provided for clustering.")
#         return {}

#     x_all_np = np.array(x_all)
#     if x_all_np.ndim == 1: # Handle case where features might be 1D lists
#         x_all_np = x_all_np.reshape(-1, 1)

#     logger.info(f"Running KMeans with n_clusters={n_clusters} on {x_all_np.shape[0]} samples...")
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10) # Explicitly set n_init
#     kmeans.fit(x_all_np)
#     cluster_centers = kmeans.cluster_centers_ # Shape (n_clusters, feature_dim)
#     cluster_assignments = kmeans.labels_     # Shape (total_samples,)

#     cluster_based_centers = {}
#     # For each original class, find the cluster center that represents it best
#     for i, orig_label in enumerate(original_labels):
#         start_idx, end_idx = class_indices[i]
#         # Get cluster assignments for samples of this class
#         assignments_for_class = cluster_assignments[start_idx:end_idx]
#         if len(assignments_for_class) == 0:
#             logger.warning(f"No cluster assignments found for class {orig_label}. Assigning zero center.")
#             cluster_based_centers[orig_label] = np.zeros_like(cluster_centers[0])
#             continue
#         # Find the most frequent cluster assignment for this class
#         most_frequent_cluster = np.bincount(assignments_for_class).argmax()
#         # Assign the center of that cluster to this original class label
#         cluster_based_centers[orig_label] = cluster_centers[most_frequent_cluster]

#     logger.info("Clustering complete.")
#     return cluster_based_centers


# --- HRRP Specific Transforms (Placeholder - Implement actual 1D transforms if needed) ---
# Example: Additive noise, time shifts, scaling etc. could be implemented here
# For now, preprocessing is handled within the HRRPDataset class.
# If using torchvision-like transforms, they need to operate on 1D tensors (C, L).

# Example placeholder (not used by default, preprocessing in dataset):
# transform_train_hrrp = transforms.Compose([
#     # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01), # Add noise
#     transforms.RandomApply([transforms.Lambda(lambda x: torch.roll(x, shifts=random.randint(-5, 5), dims=-1))], p=0.5), # Random shift
# ])
# transform_val_hrrp = transforms.Compose([]) # Usually no augmentation for validation/test


# --- Truncated Normal (Copied from original SemFew utils) ---
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