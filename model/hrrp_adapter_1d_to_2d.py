# model/hrrp_adapter_1d_to_2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class HRPPtoPseudoImage(nn.Module):
    """
    Adapts a 1D HRRP signal into a 2D pseudo-image suitable for a VLM visual encoder.
    Uses an MLP followed by Reshape.

    Args:
        hrrp_length (int): Input length L of the HRRP signal.
        input_channels (int): Number of input channels (1 for magnitude).
        output_channels (int): Target number of channels for the pseudo-image (e.g., 3).
        output_size (int): Target spatial dimension (H=W) for the pseudo-image (e.g., 224).
        intermediate_dim (int): Dimension of the intermediate MLP layer. # <--- 新增参数说明
        activation (str): Activation function for MLP ('relu', 'leaky_relu', 'tanh'). # <--- (可选) 可以添加激活函数参数
    """
    def __init__(self,
                 hrrp_length: int = 1000,
                 input_channels: int = 1,
                 output_channels: int = 3,
                 output_size: int = 224,
                 intermediate_dim: int = 2048, # <--- 添加参数并设置默认值
                 activation: str = 'relu'): # <--- (可选) 添加激活函数参数
        super().__init__()

        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        # Tanh is applied at the end, maybe not needed in intermediate layer?
        # elif activation.lower() == 'tanh':
        #      act_fn = nn.Tanh()
        else:
            logger.warning(f"Unsupported activation '{activation}' for intermediate layer. Using ReLU.")
            act_fn = nn.ReLU()

        target_elements = output_channels * output_size * output_size
        self.adapter = nn.Sequential(
            nn.Flatten(), # Flatten input (B, C, L) -> (B, C*L)
            nn.Linear(input_channels * hrrp_length, intermediate_dim), # <--- 使用参数 intermediate_dim
            act_fn, # <--- 使用选择的激活函数
            nn.Linear(intermediate_dim, target_elements), # <--- 使用参数 intermediate_dim
            nn.Tanh(), # Use Tanh to keep values in a controlled range (-1, 1)
            nn.Unflatten(1, (output_channels, output_size, output_size)) # Reshape to (B, C, H, W)
        )
        logger.info(f"Initialized HRPPtoPseudoImage (MLP->Reshape): Input=(B,{input_channels},{hrrp_length}), Intermediate={intermediate_dim}, Output=(B,{output_channels},{output_size},{output_size})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input HRRP tensor, shape (BatchSize, Channels, Length)
        Returns:
            torch.Tensor: Output pseudo-image, shape (BatchSize, output_channels, output_size, output_size)
        """
        return self.adapter(x)

# Example Usage (Remains the same)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    B = 4
    L = 1000
    C_in = 1
    C_out = 3
    H_out = 224
    W_out = 224
    INTERMEDIATE = 1024 # Example different intermediate dim

    dummy_input = torch.randn(B, C_in, L)
    # Test with default intermediate dim (2048)
    adapter_default = HRPPtoPseudoImage(hrrp_length=L, input_channels=C_in, output_channels=C_out, output_size=H_out)
    output_default = adapter_default(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (default intermediate): {output_default.shape}")

    # Test with specified intermediate dim
    adapter_custom = HRPPtoPseudoImage(hrrp_length=L, input_channels=C_in, output_channels=C_out, output_size=H_out, intermediate_dim=INTERMEDIATE)
    output_custom = adapter_custom(dummy_input)
    print(f"Output shape (intermediate={INTERMEDIATE}): {output_custom.shape}") # Expected (B, C_out, H_out, W_out)