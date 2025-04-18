import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class HRPPtoPseudoImage(nn.Module):
    """
    Adapts a 1D HRRP signal into a 2D pseudo-image suitable for a VLM visual encoder.
    Uses transposed convolutions to upsample and reshape.

    Args:
        hrrp_length (int): Input length L of the HRRP signal.
        input_channels (int): Number of input channels (1 for magnitude, 2 for complex).
        output_channels (int): Target number of channels for the pseudo-image (e.g., 3 for RGB-like).
        output_size (int): Target spatial dimension (H=W) for the pseudo-image (e.g., 224).
        intermediate_channels (list): List of channel sizes for intermediate transposed conv layers.
        kernel_size (int): Kernel size for transposed convolutions.
        stride (int): Stride for transposed convolutions.
        padding (int): Padding for transposed convolutions.
        output_padding (int): Output padding for transposed convolutions.
        activation (str): Activation function ('relu', 'leaky_relu', 'tanh').
        use_batchnorm (bool): Whether to use BatchNorm1d after transposed conv layers.
    """
    def __init__(self,
                 hrrp_length: int = 1000,
                 input_channels: int = 1,
                 output_channels: int = 3,
                 output_size: int = 224,
                 intermediate_channels: list = [64, 128, 256],
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 activation: str = 'relu',
                 use_batchnorm: bool = True):
        super().__init__()

        if activation.lower() == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'tanh':
             act_fn = nn.Tanh()
        else:
            logger.warning(f"Unsupported activation '{activation}'. Using ReLU.")
            act_fn = nn.ReLU(inplace=True)

        layers = []
        in_ch = input_channels
        current_len = hrrp_length

        # --- Upsampling Layers ---
        # Add transposed conv layers to increase spatial dimension and change channels
        all_channels = [in_ch] + intermediate_channels
        num_upsample_layers = len(intermediate_channels)

        # Calculate required upsampling factor
        # We need to reach a length L' such that L' = C_final * H * W / C_intermediate_last
        # This calculation is tricky. Let's aim for expanding length first.
        # Simplified: Estimate output length after N layers: L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
        # We want L_out close to output_size * output_size * some_factor

        # Let's use a simpler MLP -> Reshape approach first, as ConvTranspose1d sizing is complex to get right automatically
        # Option B: MLP + Reshape
        target_elements = output_channels * output_size * output_size
        self.adapter = nn.Sequential(
            # Optional: Initial 1D processing if needed
            # nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool1d(hrrp_length // 4), # Reduce length first?

            nn.Flatten(), # Flatten input (B, C, L) -> (B, C*L)
            nn.Linear(input_channels * hrrp_length, 2048), # Intermediate MLP layer
            nn.ReLU(),
            nn.Linear(2048, target_elements), # Project to target number of elements
            nn.Tanh(), # Use Tanh to keep values in a controlled range (-1, 1), similar to normalized images
            nn.Unflatten(1, (output_channels, output_size, output_size)) # Reshape to (B, C, H, W)
        )
        logger.info(f"Initialized HRPPtoPseudoImage (MLP->Reshape): Input=(B,{input_channels},{hrrp_length}), Output=(B,{output_channels},{output_size},{output_size})")

        # Option C: ConvTranspose1d (More complex to get exact size)
        # Need to carefully design layers based on input/output L, H, W
        # Example (might need adjustment):
        # layers.append(self._make_transpose_block(in_ch, intermediate_channels[0], kernel_size, stride, padding, act_fn, use_batchnorm))
        # current_len = (current_len - 1) * stride - 2 * padding + kernel_size # Rough estimate
        # ... repeat for intermediate_channels ...
        # Add final layer to get output_channels * output_size * output_size elements
        # final_len_needed = output_channels * output_size * output_size
        # final_projection_channels = ?
        # layers.append(nn.ConvTranspose1d(all_channels[-1], final_projection_channels, ...))
        # layers.append(nn.Flatten())
        # layers.append(nn.Linear(?, final_len_needed))
        # layers.append(nn.Unflatten(1, (output_channels, output_size, output_size)))
        # self.adapter = nn.Sequential(*layers)

    # Helper for Option C (ConvTranspose1d) - keep if exploring later
    # def _make_transpose_block(self, in_c, out_c, k, s, p, act_fn, use_bn):
    #     block = [nn.ConvTranspose1d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not use_bn)]
    #     if use_bn:
    #         block.append(nn.BatchNorm1d(out_c))
    #     block.append(act_fn)
    #     return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input HRRP tensor, shape (BatchSize, Channels, Length)
        Returns:
            torch.Tensor: Output pseudo-image, shape (BatchSize, output_channels, output_size, output_size)
        """
        return self.adapter(x)

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    B = 4
    L = 1000
    C_in = 1
    C_out = 3
    H_out = 224
    W_out = 224

    dummy_input = torch.randn(B, C_in, L)
    adapter = HRPPtoPseudoImage(hrrp_length=L, input_channels=C_in, output_channels=C_out, output_size=H_out)
    output = adapter(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected (B, C_out, H_out, W_out)