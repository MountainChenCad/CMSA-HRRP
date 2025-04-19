import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

class AlignmentModule(nn.Module):
    """
    (Deprecated/Unused in Adapter version)
    Maps HRRP features to the semantic space of the foundation model.
    Input: HRRP features z_H (BatchSize, hrrp_feat_dim)
    Output: Aligned HRRP features z'_H (BatchSize, semantic_dim)
    """
    def __init__(self, hrrp_feat_dim: int, semantic_dim: int, hidden_dim: int = 1024, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hrrp_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, semantic_dim) # Output matches semantic feature dimension
        )
        logger.info(f"Initialized AlignmentModule (h_A): Input={hrrp_feat_dim}, Hidden={hidden_dim}, Output={semantic_dim}")

    def forward(self, hrrp_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(hrrp_features)

class SemAlignModule(nn.Module): # Renamed from FusionModule
    """
    Reconstructs a target prototype (visual center) from visual and semantic features.
    Matches 'h' in SemFew description. Used for both CMSA-HRRP and Baseline 1DCNN+Sem.
    Input: Concatenated [z_feature, z_T] (BatchSize, feature_dim + semantic_dim)
           where z_feature is z_V for CMSA-HRRP or z_H for baseline.
    Output: Reconstructed feature (BatchSize, feature_dim)
    """
    def __init__(self, visual_dim: int, semantic_dim: int, hidden_dim: int = 4096, output_dim: int = None, drop: float = 0.2):
        super().__init__()
        # Renamed input arg for clarity, but it represents the non-semantic feature dim (z_V or z_H)
        feature_dim = visual_dim
        if output_dim is None:
            output_dim = feature_dim # Output dimension should match the target feature center (z_V or z_H)

        self.model = nn.Sequential(
            nn.Linear(feature_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop) # Redundant dropout? Model already has dropout. Maybe remove this one.

        logger.info(f"Initialized SemAlignModule (h): Input={feature_dim + semantic_dim}, Hidden={hidden_dim}, Output={output_dim}")

    def forward(self, feature: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature (torch.Tensor): Input feature (z_V or z_H). Shape (BatchSize, feature_dim)
            semantic (torch.Tensor): Semantic feature (z_T). Shape (BatchSize, semantic_dim) or (1, semantic_dim)
        Returns:
            torch.Tensor: Reconstructed feature. Shape (BatchSize, output_dim) (usually feature_dim)
        """
        # Ensure inputs are 2D (BatchSize, Dim)
        if feature.ndim > 2:
            feature = feature.view(feature.size(0), -1)
        if semantic.ndim > 2:
            semantic = semantic.view(semantic.size(0), -1)

        # Ensure semantic features are repeated if necessary (e.g., one semantic per batch of features)
        if feature.size(0) > semantic.size(0) and semantic.size(0) == 1:
             semantic = semantic.repeat(feature.size(0), 1)
        elif feature.size(0) != semantic.size(0):
             raise ValueError(f"Batch size mismatch in SemAlignModule: feature {feature.size(0)}, semantic {semantic.size(0)}")

        # Check dimension consistency before concatenation (add checks if needed)
        # input_dim = self.model[0].in_features
        # expected_feat_dim = input_dim - semantic.shape[-1] # Infer expected feat dim from layer
        # if feature.shape[-1] != expected_feat_dim:
        #      logger.error(f"Input feature dim {feature.shape[-1]} != expected {expected_feat_dim}")
        #      # Handle error

        input_concat = torch.cat((feature, semantic), dim=-1)
        fusion = self.model(input_concat)
        # fusion = self.drop(fusion) # Maybe remove redundant dropout
        reconstructed_prototype = self.fc(fusion)
        return reconstructed_prototype
