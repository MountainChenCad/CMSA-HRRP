# model/hrrp_adapter_1d_to_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional # Added Optional
import math
# Removed numpy import as it's not used here anymore

logger = logging.getLogger(__name__)
EPS = 1e-8 # Epsilon for numerical stability (though not used in this version)

# GAFTransform is no longer needed for this approach, but we keep it
# in the file if other code might import it directly.
class GAFTransform(nn.Module):
    """
    Calculates the Gramian Angular Summation Field (GASF) for a batch of 1D signals.
    Input shape: (BatchSize, 1, Length)
    Output shape: (BatchSize, 1, Length, Length)
    NOTE: This class is NOT used by the Transformer-based HRPPtoPseudoImage below.
    """
    def __init__(self, scale_range=(-1, 1), clip_val=1.0 - EPS):
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.clip_val = clip_val
        # Reduced logging noise
        # logger.info(f"Initialized GAFTransform (GASF): Scaling to [{self.scale_min}, {self.scale_max}]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (implementation remains the same, but won't be called by the adapter) ...
        min_val = torch.min(x, dim=-1, keepdim=True)[0]
        max_val = torch.max(x, dim=-1, keepdim=True)[0]
        range_val = max_val - min_val + EPS
        x_scaled_01 = (x - min_val) / range_val
        x_scaled = x_scaled_01 * (self.scale_max - self.scale_min) + self.scale_min
        x_scaled = torch.clamp(x_scaled, -self.clip_val, self.clip_val)
        phi = torch.acos(x_scaled)
        sin_phi_sq = 1 - x_scaled**2
        sin_phi = torch.sqrt(torch.clamp(sin_phi_sq, min=0.0))
        cos_cos = torch.matmul(x_scaled.transpose(-1, -2), x_scaled)
        sin_sin = torch.matmul(sin_phi.transpose(-1, -2), sin_phi)
        gasf = cos_cos - sin_sin
        return gasf.unsqueeze(1)

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's part of the state_dict but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x is expected to be (SeqLen, Batch, Dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class HRPPtoPseudoImage(nn.Module):
    """
    Adapts a 1D HRRP signal using a 1D Transformer followed by a projection
    head to produce a pseudo-image tensor compatible with CLIP's input shape.
    Maintains the __init__ signature of the previous GAF/CNN versions for compatibility,
    ignoring irrelevant parameters.

    Args:
        hrrp_length (int): Input length L of the HRRP signal. Used for pos encoding.
        input_channels (int): Number of input channels (typically 1). Used for input projection.
        gaf_size (Optional[int]): Ignored. Kept for interface compatibility.
        cnn_channels (Optional[List[int]]): Ignored. Kept for interface compatibility.
        output_channels (int): Target number of channels for the final pseudo-image (e.g., 3).
        output_size (int): Target spatial dimension (H=W) for the final pseudo-image (e.g., 224).
        kernel_size (Optional[int]): Ignored. Kept for interface compatibility.
        activation (Optional[str]): Ignored (except for final Tanh). Kept for interface compatibility.
        use_batchnorm (Optional[bool]): Ignored. Kept for interface compatibility.
    """
    def __init__(self,
                 hrrp_length: int = 1000,
                 input_channels: int = 1,
                 # --- Parameters kept for compatibility but ignored ---
                 gaf_size: Optional[int] = 64,
                 cnn_channels: Optional[List[int]] = None,
                 kernel_size: Optional[int] = 3,
                 activation: Optional[str] = 'relu', # Note: Final activation is hardcoded Tanh
                 use_batchnorm: Optional[bool] = True,
                 # --- Parameters relevant to this architecture ---
                 output_channels: int = 3,
                 output_size: int = 224):
        super().__init__()

        # Log warnings if ignored parameters are explicitly provided (not None/default)
        if gaf_size != 64: logger.warning("`gaf_size` parameter is ignored when using 1D Transformer backbone.")
        if cnn_channels is not None: logger.warning("`cnn_channels` parameter is ignored when using 1D Transformer backbone.")
        if kernel_size != 3: logger.warning("`kernel_size` parameter is ignored when using 1D Transformer backbone.")
        if activation != 'relu': logger.warning("`activation` parameter (except for final Tanh) is ignored when using 1D Transformer backbone.")
        if not use_batchnorm: logger.warning("`use_batchnorm` parameter is ignored when using 1D Transformer backbone.")

        self.hrrp_length = hrrp_length
        self.output_channels = output_channels
        self.output_size = output_size

        # --- Hardcoded Transformer Hyperparameters ---
        # These are fixed as they cannot be passed via the current __init__ signature
        d_model = 256       # Transformer embedding dimension
        nhead = 4           # Number of attention heads
        d_hid = 512         # Dimension of the feedforward network model in nn.TransformerEncoderLayer
        nlayers = 3         # Number of nn.TransformerEncoderLayer layers
        dropout = 0.1       # Dropout probability
        # -------------------------------------------

        # 1. Input Embedding: Project input channels to d_model
        # Input shape: (B, C, L) -> (B, L, C) -> (B, L, d_model)
        self.input_proj = nn.Linear(input_channels, d_model)

        # 2. Positional Encoding
        # Needs max_len >= hrrp_length
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max(hrrp_length, 5000))

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True) # Use batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # 4. Projection Head
        # Output of Transformer (mean pooled): (B, d_model)
        # Target flattened size: output_channels * output_size * output_size
        target_elements = output_channels * output_size * output_size
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model), # Normalize before projection
            nn.Linear(d_model, target_elements),
            nn.Tanh() # Use Tanh like previous adapters
        )

        logger.info(f"Initialized HRPPtoPseudoImage (1D Transformer Backbone): Input=(B,{input_channels},{hrrp_length}) -> "
                    f"InputProj({d_model}) -> PosEnc -> TransformerEncoder(L={nlayers}, H={nhead}, D={d_model}) -> "
                    f"MeanPool -> ProjectionHead -> Tanh -> Unflatten({output_channels}x{output_size}x{output_size}). "
                    f"Ignoring gaf_size, cnn_channels, kernel_size, activation (partially), use_batchnorm parameters.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input HRRP tensor, shape (BatchSize, Channels, Length)
        Returns:
            torch.Tensor: Output pseudo-image, shape (BatchSize, output_channels, output_size, output_size)
        """
        # Ensure input has the expected channel dimension for input_proj
        if x.shape[1] != self.input_proj.in_features:
             # This case should ideally be handled by input_channels param, but good to check.
             logger.warning(f"Input tensor channel dim {x.shape[1]} doesn't match expected {self.input_proj.in_features}. Taking first {self.input_proj.in_features} channels.")
             x = x[:, :self.input_proj.in_features, :]


        # Reshape for Transformer: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1) # Now (B, L, C)

        # 1. Input Projection: (B, L, C) -> (B, L, d_model)
        x = self.input_proj(x)

        # 2. Add Positional Encoding
        # TransformerEncoderLayer expects (SeqLen, Batch, Dim) if batch_first=False
        # or (Batch, SeqLen, Dim) if batch_first=True.
        # PositionalEncoding expects (SeqLen, Batch, Dim). Let's adjust.
        # Input to pos_encoder: (L, B, d_model)
        # Output from pos_encoder: (L, B, d_model)
        # Input to transformer (batch_first=True): (B, L, d_model)

        # We'll adapt PositionalEncoding or adjust the flow. Easier to adjust flow:
        # Add PE in (B, L, d_model) format if PE class is modified or just add directly.
        # Let's stick to standard PE needing (L, B, D) input for now.
        # x = x.permute(1, 0, 2) # (L, B, d_model)
        # x = self.pos_encoder(x)
        # x = x.permute(1, 0, 2) # (B, L, d_model) - Back for batch_first=True encoder

        # Simpler: If PositionalEncoding is modified for batch_first=True:
        # self.pos_encoder needs modification to accept (B, L, D)
        # OR manually add PE assuming pe shape is (1, max_len, d_model) or (max_len, d_model)
        # Let's assume self.pos_encoder.pe is (max_len, 1, d_model) as defined
        pe = self.pos_encoder.pe[:x.size(1), 0, :].unsqueeze(0) # Shape (1, L, d_model)
        x = x + pe # Add positional encoding (broadcasts over batch dim B)
        x = self.pos_encoder.dropout(x) # Apply dropout after adding PE

        # 3. Pass through Transformer Encoder
        # Input shape (B, L, d_model) because batch_first=True
        transformer_output = self.transformer_encoder(x) # Output shape (B, L, d_model)

        # 4. Aggregate sequence features (e.g., mean pooling)
        # Take the mean over the sequence length dimension (L)
        aggregated_features = transformer_output.mean(dim=1) # Shape (B, d_model)

        # 5. Projection Head
        projected_features = self.projection_head(aggregated_features) # Shape (B, target_elements)

        # 6. Unflatten to pseudo-image shape
        output_image = projected_features.view(x.size(0), # Batch size B
                                               self.output_channels,
                                               self.output_size,
                                               self.output_size)
        # Shape: (B, output_channels, output_size, output_size)

        return output_image

# Example Usage (Demonstrating compatibility)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    B = 4
    L = 1000 # hrrp_length
    C_in = 1 # input_channels
    CLIP_SIZE = 224 # output_size
    CLIP_CHANNELS = 3 # output_channels

    dummy_input = torch.randn(B, C_in, L)

    # --- Test the 1D Transformer Adapter using the old signature ---
    # We pass the old arguments; irrelevant ones will be ignored.
    transformer_adapter = HRPPtoPseudoImage(
        hrrp_length=L,
        input_channels=C_in,
        gaf_size=128,          # <--- Ignored
        cnn_channels=[16, 32], # <--- Ignored
        kernel_size=5,         # <--- Ignored
        activation='leaky_relu',# <--- Ignored (except final Tanh)
        use_batchnorm=False,   # <--- Ignored
        output_channels=CLIP_CHANNELS,
        output_size=CLIP_SIZE
    )
    output_transformer = transformer_adapter(dummy_input)
    print(f"\n--- 1D Transformer Adapter (Compatible Signature) ---")
    print(f"Input shape: {dummy_input.shape}")
    # Expected: [B, CLIP_CHANNELS, CLIP_SIZE, CLIP_SIZE]
    print(f"Output shape: {output_transformer.shape}")
    assert output_transformer.shape == (B, CLIP_CHANNELS, CLIP_SIZE, CLIP_SIZE)

    # --- Compare with the original MLP Adapter (for reference) ---
    class HRPPtoPseudoImageMLP(nn.Module): # Keep definition for comparison
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
        intermediate_dim=2048
    )
    # output_mlp = mlp_adapter(dummy_input)
    # print(f"\n--- Original MLP Adapter ---")
    # print(f"Input shape: {dummy_input.shape}")
    # print(f"Output shape: {output_mlp.shape}")

    # Parameter Count Comparison
    params_transformer = sum(p.numel() for p in transformer_adapter.parameters() if p.requires_grad)
    params_mlp = sum(p.numel() for p in mlp_adapter.parameters() if p.requires_grad)
    print(f"\nParameter Count:")
    print(f"1D Transformer Adapter: {params_transformer:,}")
    print(f"MLP Adapter           : {params_mlp:,}")
    print(f"(Note: Transformer params depend on hardcoded d_model, nhead, nlayers etc.)")
