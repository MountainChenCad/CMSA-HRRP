# model/hrrp_adapter_1d_to_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
import math
import numpy as np # Added for clipping

logger = logging.getLogger(__name__)
EPS = 1e-8 # Epsilon for numerical stability

class GAFTransform(nn.Module):
    """
    Calculates the Gramian Angular Summation Field (GASF) for a batch of 1D signals.
    Input shape: (BatchSize, 1, Length)
    Output shape: (BatchSize, 1, Length, Length)
    """
    def __init__(self, scale_range=(-1, 1), clip_val=1.0 - EPS):
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.clip_val = clip_val
        logger.info(f"Initialized GAFTransform (GASF): Scaling to [{self.scale_min}, {self.scale_max}]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, L)
        # 1. Scale the time series to specified range [-1, 1] for acos
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        range_val = max_val - min_val + EPS # Add epsilon to avoid division by zero

        # Scale to [0, 1] first
        x_scaled_01 = (x - min_val) / range_val
        # Scale to [scale_min, scale_max]
        x_scaled = x_scaled_01 * (self.scale_max - self.scale_min) + self.scale_min

        # Clip values to be within [-clip_val, clip_val] for numerical stability with acos
        x_scaled = torch.clamp(x_scaled, -self.clip_val, self.clip_val)

        # 2. Calculate angles using arccos
        phi = torch.acos(x_scaled) # Shape: (B, 1, L)

        # 3. Calculate GASF: cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
        # cos(phi) is just x_scaled
        # sin(phi) = sqrt(1 - cos(phi)^2) = sqrt(1 - x_scaled^2)
        sin_phi_sq = 1 - x_scaled**2
        # Clamp sin_phi_sq to be non-negative before sqrt
        sin_phi = torch.sqrt(torch.clamp(sin_phi_sq, min=0.0)) # Shape: (B, 1, L)

        # Use matrix multiplication for efficiency
        # cos(phi_i)cos(phi_j) -> outer product of x_scaled with itself
        cos_cos = torch.matmul(x_scaled.transpose(-1, -2), x_scaled) # (B, L, 1) * (B, 1, L) -> (B, L, L)
        # sin(phi_i)sin(phi_j) -> outer product of sin_phi with itself
        sin_sin = torch.matmul(sin_phi.transpose(-1, -2), sin_phi) # (B, L, 1) * (B, 1, L) -> (B, L, L)

        # GASF = cos_cos - sin_sin
        gasf = cos_cos - sin_sin # Shape: (B, L, L)

        # Add channel dimension back: (B, 1, L, L)
        return gasf.unsqueeze(1)


class HRPPtoPseudoImage(nn.Module):
    """
    Adapts a 1D HRRP signal to a 2D pseudo-image via GAF transformation
    followed by a CNN feature extractor and resizing.

    Args:
        hrrp_length (int): Input length L of the HRRP signal.
        input_channels (int): Number of input channels (usually 1 for magnitude).
        gaf_size (int): The size (L') to resize the 1D signal to before GAF. GAF image will be L'xL'.
        cnn_channels (List[int]): Output channels for each CNN block.
        output_channels (int): Target number of channels for the final pseudo-image (e.g., 3).
        output_size (int): Target spatial dimension (H=W) for the final pseudo-image (e.g., 224).
        kernel_size (int): Kernel size for CNN layers.
        activation (str): Activation function for CNN ('relu', 'leaky_relu').
        use_batchnorm (bool): Whether to use BatchNorm2d in CNN blocks.
    """
    def __init__(self,
                 hrrp_length: int = 1000,
                 input_channels: int = 1,
                 gaf_size: int = 64, # Resize 1D signal to this length first
                 cnn_channels: List[int] = [16, 32, 64], # Example CNN channels
                 output_channels: int = 3,
                 output_size: int = 224,
                 kernel_size: int = 3,
                 activation: str = 'relu',
                 use_batchnorm: bool = True):
        super().__init__()

        self.hrrp_length = hrrp_length
        self.gaf_size = gaf_size

        # 1. GAF Transformation Module
        self.gaf_transform = GAFTransform()

        # 2. CNN Feature Extractor
        if activation.lower() == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1, inplace=True)
        else:
            logger.warning(f"Unsupported activation '{activation}'. Using ReLU.")
            act_fn = nn.ReLU(inplace=True)

        cnn_blocks = []
        in_c = 1 # GAF image has 1 channel
        padding = kernel_size // 2
        for out_c in cnn_channels:
            block = [
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, bias=not use_batchnorm)
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm2d(out_c))
            block.append(act_fn)
            # Optional: Add pooling to reduce size if needed, depends on gaf_size
            # block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            cnn_blocks.append(nn.Sequential(*block))
            in_c = out_c

        self.cnn_extractor = nn.Sequential(*cnn_blocks)
        final_cnn_channels = cnn_channels[-1] if cnn_channels else 1

        # 3. Output Adaptation Layers
        # Resize spatial dimensions to target output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_size, output_size))
        # Project channels to the desired output_channels (e.g., 3 for CLIP)
        self.channel_adjust_conv = nn.Conv2d(final_cnn_channels, output_channels, kernel_size=1)
        # Tanh activation similar to the original MLP adapter
        self.final_activation = nn.Tanh()

        logger.info(f"Initialized HRPPtoPseudoImage: Input=(B,{input_channels},{hrrp_length}) -> "
                    f"Resize1D({gaf_size}) -> GAF({gaf_size}x{gaf_size}) -> "
                    f"CNN(Channels:{cnn_channels}) -> AdaptivePool({output_size}x{output_size}) -> "
                    f"Conv1x1(Channels:{output_channels}) -> Tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input HRRP tensor, shape (BatchSize, Channels=1, Length)
        Returns:
            torch.Tensor: Output pseudo-image, shape (BatchSize, output_channels, output_size, output_size)
        """
        # Ensure input has 1 channel if specified
        if x.shape[1] != 1:
            logger.warning(f"Input tensor has {x.shape[1]} channels, expected 1. Taking the first channel.")
            x = x[:, 0:1, :]

        # 1. Resize 1D signal (optional, but helps manage GAF size)
        if self.hrrp_length != self.gaf_size:
            x_resized = F.interpolate(x, size=self.gaf_size, mode='linear', align_corners=False)
        else:
            x_resized = x
        # Shape: (B, 1, gaf_size)

        # 2. GAF Transformation
        gaf_image = self.gaf_transform(x_resized) # Shape: (B, 1, gaf_size, gaf_size)

        # 3. CNN Feature Extraction
        cnn_features = self.cnn_extractor(gaf_image) # Shape: (B, final_cnn_channels, H', W')

        # 4. Output Adaptation
        pooled_features = self.adaptive_pool(cnn_features) # Shape: (B, final_cnn_channels, output_size, output_size)
        adjusted_features = self.channel_adjust_conv(pooled_features) # Shape: (B, output_channels, output_size, output_size)
        output_image = self.final_activation(adjusted_features) # Shape: (B, output_channels, output_size, output_size)

        return output_image

# Example Usage (replace the old HRPPtoPseudoImage class with HRPPtoPseudoImage)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    B = 4
    L = 1000
    C_in = 1
    CLIP_SIZE = 224
    CLIP_CHANNELS = 3

    dummy_input = torch.randn(B, C_in, L)

    # --- Test the new GAF+CNN Adapter ---
    # Parameters can be adjusted
    gaf_adapter = HRPPtoPseudoImage(
        hrrp_length=L,
        input_channels=C_in,
        gaf_size=64, # Size of the GAF image (e.g., 64x64)
        cnn_channels=[16, 32, 64], # Output channels of CNN layers
        output_channels=CLIP_CHANNELS, # Target channels (e.g., 3)
        output_size=CLIP_SIZE # Target size (e.g., 224)
    )
    output_gaf = gaf_adapter(dummy_input)
    print(f"\n--- GAF+CNN Adapter ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_gaf.shape}") # Expected: [B, 3, 224, 224]

    # --- Compare with the original MLP Adapter (for reference) ---
    # Re-define the original MLP adapter for comparison context
    class HRPPtoPseudoImageMLP(nn.Module):
        def __init__(self, hrrp_length=1000, input_channels=1, output_channels=3, output_size=224, intermediate_dim=2048, activation='relu'):
            super().__init__()
            if activation.lower() == 'relu': act_fn = nn.ReLU()
            elif activation.lower() == 'leaky_relu': act_fn = nn.LeakyReLU(0.2)
            else: act_fn = nn.ReLU()
            target_elements = output_channels * output_size * output_size
            self.adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels * hrrp_length, intermediate_dim),
                act_fn,
                nn.Linear(intermediate_dim, target_elements),
                nn.Tanh(),
                nn.Unflatten(1, (output_channels, output_size, output_size))
            )
        def forward(self, x): return self.adapter(x)

    mlp_adapter = HRPPtoPseudoImageMLP(
        hrrp_length=L,
        input_channels=C_in,
        output_channels=CLIP_CHANNELS,
        output_size=CLIP_SIZE,
        intermediate_dim=2048 # Original intermediate dim
    )
    output_mlp = mlp_adapter(dummy_input)
    print(f"\n--- Original MLP Adapter ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_mlp.shape}") # Expected: [B, 3, 224, 224]

    # Parameter Count Comparison
    params_gaf = sum(p.numel() for p in gaf_adapter.parameters() if p.requires_grad)
    params_mlp = sum(p.numel() for p in mlp_adapter.parameters() if p.requires_grad)
    print(f"\nParameter Count:")
    print(f"GAF+CNN Adapter: {params_gaf:,}")
    print(f"MLP Adapter    : {params_mlp:,}")
    print(f"GAF+CNN is significantly more lightweight: {params_mlp / params_gaf:.1f}x smaller (approx)")
