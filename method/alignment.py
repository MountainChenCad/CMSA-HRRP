import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FusionModule(nn.Module):
    """
    Fuses aligned HRRP features and semantic features to reconstruct a prototype.
    Similar to SemAlign in SemFew.
    Input: Concatenated [z'_H, z_T] (BatchSize, aligned_hrrp_dim + semantic_dim)
    Output: Reconstructed prototype feature (BatchSize, aligned_hrrp_dim or semantic_dim)
            Let's output semantic_dim to align with the target in training.
    """
    def __init__(self, aligned_hrrp_dim: int, semantic_dim: int, hidden_dim: int = 4096, output_dim: int = None, drop: float = 0.2):
        super().__init__()
        if output_dim is None:
            output_dim = semantic_dim # Default to outputting in semantic space dimension

        self.model = nn.Sequential(
            nn.Linear(aligned_hrrp_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        logger.info(f"Initialized FusionModule (h_F): Input={aligned_hrrp_dim + semantic_dim}, Hidden={hidden_dim}, Output={output_dim}")

    def forward(self, aligned_hrrp: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 2D (BatchSize, Dim)
        if aligned_hrrp.ndim > 2:
            aligned_hrrp = aligned_hrrp.view(aligned_hrrp.size(0), -1)
        if semantic.ndim > 2:
            semantic = semantic.view(semantic.size(0), -1)

        # Ensure semantic features are repeated if necessary (e.g., matching batch size)
        if aligned_hrrp.size(0) > semantic.size(0) and semantic.size(0) == 1:
             semantic = semantic.repeat(aligned_hrrp.size(0), 1)
        elif aligned_hrrp.size(0) != semantic.size(0):
             raise ValueError(f"Batch size mismatch in FusionModule: aligned_hrrp {aligned_hrrp.size(0)}, semantic {semantic.size(0)}")


        input_concat = torch.cat((aligned_hrrp, semantic), dim=-1)
        fusion = self.model(input_concat)
        fusion = self.drop(fusion)
        reconstructed_prototype = self.fc(fusion)
        return reconstructed_prototype
