import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

class AlignmentModule(nn.Module):
    """
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
    Matches 'h' in SemFew description.
    Input: Concatenated [z_V, z_T] (BatchSize, visual_dim + semantic_dim)
    Output: Reconstructed visual center feature (BatchSize, visual_dim)
    """
    def __init__(self, visual_dim: int, semantic_dim: int, hidden_dim: int = 4096, output_dim: int = None, drop: float = 0.2):
        super().__init__()
        if output_dim is None:
            output_dim = visual_dim # Output dimension should match the target visual center C_y

        self.model = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        logger.info(f"Initialized SemAlignModule (h): Input={visual_dim + semantic_dim}, Hidden={hidden_dim}, Output={output_dim}")

    def forward(self, visual_feature: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D (BatchSize, Dim)
        if visual_feature.ndim > 2:
            visual_feature = visual_feature.view(visual_feature.size(0), -1)
        if semantic.ndim > 2:
            semantic = semantic.view(semantic.size(0), -1)

        # Ensure semantic features are repeated if necessary
        if visual_feature.size(0) > semantic.size(0) and semantic.size(0) == 1:
             semantic = semantic.repeat(visual_feature.size(0), 1)
        elif visual_feature.size(0) != semantic.size(0):
             raise ValueError(f"Batch size mismatch in SemAlignModule: visual {visual_feature.size(0)}, semantic {semantic.size(0)}")

        # Check dimension consistency before concatenation
        # Assuming visual_dim and semantic_dim were passed correctly during init
        # Add explicit checks if needed based on config

        input_concat = torch.cat((visual_feature, semantic), dim=-1)
        fusion = self.model(input_concat)
        fusion = self.drop(fusion)
        reconstructed_prototype = self.fc(fusion)
        return reconstructed_prototype
