import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class HRRPEncoder(nn.Module):
    """
    A simple 1D CNN based encoder for HRRP signals.
    Input: HRRP tensor (BatchSize, Channels, Length) - Assuming Channels=1 or 2 (complex)
    Output: Feature vector (BatchSize, output_dim)
    """
    def __init__(self,
                 input_channels: int = 1, # 1 for magnitude/real, 2 for complex (real/imag)
                 output_dim: int = 512,
                 layers: list = [64, 128, 256, 512], # Channels in each conv block
                 kernel_size: int = 7,
                 stride: int = 1,
                 padding: int = 3,
                 pool_kernel: int = 3,
                 pool_stride: int = 2,
                 use_batchnorm: bool = True,
                 activation: str = 'relu',
                 final_pool: str = 'adaptive_avg'): # 'adaptive_avg', 'adaptive_max', 'flatten'
        super().__init__()

        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1)
        else:
            logger.warning(f"Unsupported activation '{activation}'. Using ReLU.")
            act_fn = nn.ReLU()

        conv_blocks = []
        in_channels = input_channels
        current_length = -1 # Track length if needed for flatten, but adaptive is preferred
        for i, out_channels in enumerate(layers):
            block = [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_channels))
            block.append(act_fn)
            block.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_kernel // 2))
            conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels
            # Rough length calculation (if needed for flatten)
            # if current_length != -1:
            #     conv_out_len = (current_length + 2 * padding - kernel_size) // stride + 1
            #     current_length = (conv_out_len + 2 * (pool_kernel // 2) - pool_kernel) // pool_stride + 1

        self.encoder = nn.Sequential(*conv_blocks)

        self.final_pool_type = final_pool
        if final_pool == 'adaptive_avg':
            self.final_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(in_channels, output_dim) # Add a final linear layer
        elif final_pool == 'adaptive_max':
            self.final_pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(in_channels, output_dim)
        elif final_pool == 'flatten':
            self.final_pool = nn.Flatten(1) # Flatten from channel dim
            # Calculate flattened size dynamically (requires passing input_length or using a dummy forward pass)
            # This is complex, adaptive pooling is recommended.
            # Example placeholder (likely incorrect without dynamic calculation):
            # dummy_input_len = 1000 # Example
            # calculated_length = self._get_conv_output_length(dummy_input_len, layers, kernel_size, stride, padding, pool_kernel, pool_stride)
            # self.fc = nn.Linear(in_channels * calculated_length, output_dim)
            logger.warning("Flatten pooling selected. FC layer input size might be incorrect. Use adaptive pooling for robustness.")
            self.fc = nn.Linear(in_channels, output_dim) # Placeholder, likely needs adjustment
        else:
             logger.warning(f"Unsupported final_pool '{final_pool}'. Using AdaptiveAvgPool1d.")
             self.final_pool = nn.AdaptiveAvgPool1d(1)
             self.fc = nn.Linear(in_channels, output_dim)

        logger.info(f"Initialized HRRPEncoder: InChannels={input_channels}, Layers={layers}, OutDim={output_dim}, FinalPool={self.final_pool_type}")
        # Initialize weights (optional but good practice)
        self._initialize_weights()

    # Helper to calculate output length (if needed for flatten)
    def _get_conv_output_length(self, L_in, layers, kernel, stride, padding, pool_k, pool_s):
         L_out = L_in
         for _ in layers:
              L_conv = (L_out + 2 * padding - kernel) // stride + 1
              L_out = (L_conv + 2 * (pool_k // 2) - pool_k) // pool_s + 1
         return L_out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input HRRP tensor, shape (BatchSize, Channels, Length)
        Returns:
            torch.Tensor: Output feature vector, shape (BatchSize, output_dim)
        """
        x = self.encoder(x) # (B, C_last, L_out)
        x = self.final_pool(x) # (B, C_last, 1) for adaptive pool, (B, C_last * L_out) for flatten
        if self.final_pool_type != 'flatten':
             x = torch.flatten(x, 1) # Flatten after pooling: (B, C_last)
        # Else: x is already flattened by nn.Flatten(1)
        x = self.fc(x) # Apply final linear layer
        return x

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Example: Input magnitude HRRP of length 1000
    dummy_input = torch.randn(16, 1, 1000) # (BatchSize, Channels, Length)
    encoder = HRRPEncoder(input_channels=1, output_dim=512)
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Expected (16, 512)

    # Example: Input complex HRRP (real/imag stacked) of length 500
    dummy_complex_input = torch.randn(16, 2, 500)
    encoder_complex = HRRPEncoder(input_channels=2, output_dim=640)
    output_complex = encoder_complex(dummy_complex_input)
    print(f"Complex Input shape: {dummy_complex_input.shape}")
    print(f"Complex Output shape: {output_complex.shape}") # Expected (16, 640)
