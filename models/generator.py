"""
Conditional Generator for PxGAN
Generates synthetic telemetry pixel grids conditioned on metadata
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator that produces telemetry pixel grids

    Architecture:
        - Concatenates latent noise z with conditional vector
        - Fully connected layers to expand dimensionality
        - Transposed convolutions to upsample to target image size
        - Outputs multi-channel pixel grid

    Input:
        - z: latent noise vector (B, z_dim)
        - cond: conditional metadata (B, cond_dim)

    Output:
        - image: generated telemetry grid (B, out_channels, H, W)
    """

    def __init__(self,
                 z_dim: int = 128,
                 cond_dim: int = 16,
                 out_channels: int = 8,
                 img_size: int = 32,
                 hidden_dims: Tuple[int, ...] = (1024, 512),
                 use_spectral_norm: bool = False):
        """
        Args:
            z_dim: Latent noise dimension
            cond_dim: Conditional input dimension
            out_channels: Number of output channels (feature groups)
            img_size: Output image size (assumes square images)
            hidden_dims: Hidden layer dimensions for FC layers
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()

        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.out_channels = out_channels
        self.img_size = img_size

        # Input dimension after concatenating z and cond
        input_dim = z_dim + cond_dim

        # Fully connected layers to expand latent code
        fc_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim

        # Final FC layer to reshape for conv layers
        # Start with 8x8 feature maps with 128 channels
        self.init_size = 8
        self.init_channels = 128
        fc_layers.append(nn.Linear(prev_dim, self.init_channels * self.init_size * self.init_size))

        self.fc = nn.Sequential(*fc_layers)

        # Transposed convolution layers to upsample
        # 8x8 -> 16x16 -> 32x32 (default)
        conv_layers = []

        # Calculate number of upsampling layers needed
        current_size = self.init_size
        in_channels = self.init_channels

        while current_size < img_size:
            out_ch = max(in_channels // 2, 32)

            if use_spectral_norm:
                conv = nn.utils.spectral_norm(
                    nn.ConvTranspose2d(in_channels, out_ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                conv = nn.ConvTranspose2d(in_channels, out_ch, kernel_size=4, stride=2, padding=1)

            conv_layers.extend([
                conv,
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])

            in_channels = out_ch
            current_size *= 2

        # Final layer to output channels with Tanh activation
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        conv_layers.append(final_conv)
        conv_layers.append(nn.Tanh())

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            z: Latent noise (B, z_dim)
            cond: Conditional metadata (B, cond_dim)

        Returns:
            Generated images (B, out_channels, img_size, img_size)
        """
        # Concatenate z and cond
        x = torch.cat([z, cond], dim=1)  # (B, z_dim + cond_dim)

        # Fully connected layers
        x = self.fc(x)  # (B, init_channels * init_size * init_size)

        # Reshape for conv layers
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)  # (B, C, H, W)

        # Transposed convolutions
        x = self.conv(x)  # (B, out_channels, img_size, img_size)

        return x

    def sample(self, num_samples: int, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate samples

        Args:
            num_samples: Number of samples to generate
            cond: Conditional metadata (num_samples, cond_dim)
            device: Device to generate on

        Returns:
            Generated images (num_samples, out_channels, img_size, img_size)
        """
        z = torch.randn(num_samples, self.z_dim).to(device)

        with torch.no_grad():
            samples = self.forward(z, cond)

        return samples


def test_generator():
    """Test generator architecture"""
    batch_size = 16
    z_dim = 128
    cond_dim = 16
    out_channels = 8
    img_size = 32

    G = ConditionalGenerator(
        z_dim=z_dim,
        cond_dim=cond_dim,
        out_channels=out_channels,
        img_size=img_size
    )

    z = torch.randn(batch_size, z_dim)
    cond = torch.randn(batch_size, cond_dim)

    output = G(z, cond)

    print(f"Generator test:")
    print(f"  Input z: {z.shape}")
    print(f"  Input cond: {cond.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in G.parameters()):,}")

    assert output.shape == (batch_size, out_channels, img_size, img_size)
    assert output.min() >= -1.0 and output.max() <= 1.0  # Tanh output

    print("  ✓ Generator test passed!")


if __name__ == '__main__':
    test_generator()
