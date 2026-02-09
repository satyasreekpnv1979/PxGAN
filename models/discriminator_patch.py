"""
PatchGAN Discriminator for PxGAN
Outputs patch-level real/fake scores for spatial anomaly detection
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator

    Architecture:
        - Convolutional layers with stride=2 for downsampling
        - Spectral normalization for stability
        - LeakyReLU activations
        - Outputs spatial map of patch scores (not single scalar)

    Input:
        - image: telemetry pixel grid (B, in_channels, H, W)

    Output:
        - patch_scores: patch-level real/fake scores (B, 1, H', W')
    """

    def __init__(self,
                 in_channels: int = 8,
                 ndf: int = 64,
                 n_layers: int = 3,
                 use_spectral_norm: bool = True):
        """
        Args:
            in_channels: Number of input channels (feature groups)
            ndf: Base number of discriminator filters
            n_layers: Number of conv layers
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers

        # Build convolutional layers
        layers = []

        # First layer: no normalization
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
            )
        else:
            conv = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)

        layers.extend([
            conv,
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Middle layers
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base channels

            if use_spectral_norm:
                conv = nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=4, stride=2, padding=1)
                )
            else:
                conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=4, stride=2, padding=1)

            layers.extend([
                conv,
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # Penultimate layer: stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=1, padding=1)
            )
        else:
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=4, stride=1, padding=1)

        layers.extend([
            conv,
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Final layer: output patch scores (no activation)
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        else:
            final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)

        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input images (B, in_channels, H, W)

        Returns:
            Patch scores (B, 1, H', W') - raw logits, no sigmoid
        """
        return self.model(x)

    def get_intermediate_features(self, x: torch.Tensor, layer_indices: list = None):
        """
        Extract intermediate features for feature matching loss

        Args:
            x: Input images
            layer_indices: Which layers to extract (None = all conv layers)

        Returns:
            List of feature maps
        """
        features = []
        current_x = x

        if layer_indices is None:
            # Extract from all conv layers
            layer_indices = [i for i, m in enumerate(self.model) if isinstance(m, nn.Conv2d)]

        for i, layer in enumerate(self.model):
            current_x = layer(current_x)
            if i in layer_indices:
                features.append(current_x)

        return features


class MultiScalePatchDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator

    Uses multiple discriminators at different scales for better quality
    """

    def __init__(self,
                 in_channels: int = 8,
                 ndf: int = 64,
                 num_scales: int = 2,
                 use_spectral_norm: bool = True):
        """
        Args:
            in_channels: Number of input channels
            ndf: Base number of filters
            num_scales: Number of scales (discriminators)
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()

        self.num_scales = num_scales

        # Create discriminators at different scales
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, ndf, use_spectral_norm=use_spectral_norm)
            for _ in range(num_scales)
        ])

        # Downsampling for multi-scale inputs
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass at multiple scales

        Args:
            x: Input images (B, C, H, W)

        Returns:
            List of patch scores at different scales
        """
        outputs = []
        current_scale = x

        for disc in self.discriminators:
            outputs.append(disc(current_scale))
            current_scale = self.downsample(current_scale)

        return outputs


def test_patch_discriminator():
    """Test PatchGAN discriminator"""
    batch_size = 16
    in_channels = 8
    img_size = 32

    D = PatchDiscriminator(in_channels=in_channels, ndf=64, n_layers=3)

    x = torch.randn(batch_size, in_channels, img_size, img_size)
    output = D(x)

    print(f"PatchGAN Discriminator test:")
    print(f"  Input: {x.shape}")
    print(f"  Output (patch scores): {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in D.parameters()):,}")

    # Test intermediate features
    features = D.get_intermediate_features(x)
    print(f"  Num intermediate features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"    Feature {i}: {feat.shape}")

    print("  ✓ PatchGAN test passed!")


def test_multiscale_discriminator():
    """Test multi-scale discriminator"""
    batch_size = 8
    in_channels = 8
    img_size = 32

    D_multi = MultiScalePatchDiscriminator(in_channels=in_channels, num_scales=2)

    x = torch.randn(batch_size, in_channels, img_size, img_size)
    outputs = D_multi(x)

    print(f"\nMulti-scale PatchGAN test:")
    print(f"  Input: {x.shape}")
    print(f"  Num scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"    Scale {i} output: {out.shape}")

    print("  ✓ Multi-scale test passed!")


if __name__ == '__main__':
    test_patch_discriminator()
    test_multiscale_discriminator()
