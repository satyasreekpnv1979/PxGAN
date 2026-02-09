"""
Sequence Critic for PxGAN
Detects temporal inconsistencies in sequences of telemetry windows
"""

import torch
import torch.nn as nn


class SequenceCritic(nn.Module):
    """
    Sequence Critic for temporal coherence

    Architecture:
        - CNN encoder per frame to extract spatial features
        - LSTM to model temporal dependencies
        - Final classifier for real/fake sequence classification

    Input:
        - sequence: (B, seq_len, C, H, W) - sequence of telemetry grids

    Output:
        - score: scalar real/fake score for the entire sequence
    """

    def __init__(self,
                 in_channels: int = 8,
                 seq_len: int = 5,
                 cnn_embed_dim: int = 256,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.3,
                 use_spectral_norm: bool = True):
        """
        Args:
            in_channels: Number of input channels per frame
            seq_len: Sequence length (number of consecutive windows)
            cnn_embed_dim: Output dimension of CNN encoder
            lstm_hidden: LSTM hidden state dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            use_spectral_norm: Whether to use spectral normalization in CNN
        """
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len
        self.cnn_embed_dim = cnn_embed_dim
        self.lstm_hidden = lstm_hidden

        # CNN encoder for each frame
        # Input: (C, H, W) -> Output: (cnn_embed_dim,)
        cnn_layers = []

        # Conv blocks
        channels = [in_channels, 64, 128, 256]
        for i in range(len(channels) - 1):
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1)
                )
            else:
                conv = nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1)

            cnn_layers.extend([
                conv,
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # Global average pooling
        cnn_layers.append(nn.AdaptiveAvgPool2d(1))

        self.cnn_encoder = nn.Sequential(*cnn_layers)

        # FC layer to match cnn_embed_dim
        self.fc_embed = nn.Linear(channels[-1], cnn_embed_dim)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            sequence: (B, seq_len, C, H, W) - sequence of frames

        Returns:
            score: (B, 1) - real/fake score for sequence
        """
        B, T, C, H, W = sequence.shape

        # Encode each frame
        # Reshape to (B*T, C, H, W) for batch processing
        frames = sequence.view(B * T, C, H, W)

        # CNN encoding
        cnn_features = self.cnn_encoder(frames)  # (B*T, channels[-1], 1, 1)
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)  # (B*T, channels[-1])

        # Project to embedding dimension
        embeddings = self.fc_embed(cnn_features)  # (B*T, cnn_embed_dim)

        # Reshape to sequence: (B, T, cnn_embed_dim)
        embeddings = embeddings.view(B, T, self.cnn_embed_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embeddings)  # lstm_out: (B, T, lstm_hidden)

        # Use final hidden state for classification
        final_hidden = h_n[-1]  # (B, lstm_hidden)

        # Classify
        score = self.classifier(final_hidden)  # (B, 1)

        return score

    def get_temporal_features(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features (LSTM hidden states) for feature matching

        Args:
            sequence: (B, seq_len, C, H, W)

        Returns:
            LSTM output: (B, seq_len, lstm_hidden)
        """
        B, T, C, H, W = sequence.shape

        # Encode frames
        frames = sequence.view(B * T, C, H, W)
        cnn_features = self.cnn_encoder(frames).squeeze(-1).squeeze(-1)
        embeddings = self.fc_embed(cnn_features).view(B, T, self.cnn_embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(embeddings)

        return lstm_out


class SimplifiedSequenceCritic(nn.Module):
    """
    Simplified Sequence Critic using 3D convolutions

    Processes spatiotemporal data directly with 3D convolutions
    """

    def __init__(self,
                 in_channels: int = 8,
                 seq_len: int = 5,
                 base_filters: int = 32):
        """
        Args:
            in_channels: Number of channels per frame
            seq_len: Sequence length
            base_filters: Base number of filters
        """
        super().__init__()

        self.in_channels = in_channels
        self.seq_len = seq_len

        # 3D Conv layers: (C, T, H, W) input
        self.conv3d = nn.Sequential(
            # (in_channels, seq_len, 32, 32) -> (32, seq_len//2, 16, 16)
            nn.Conv3d(in_channels, base_filters, kernel_size=(3, 4, 4),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters),
            nn.LeakyReLU(0.2),

            # (32, seq_len//2, 16, 16) -> (64, seq_len//4, 8, 8)
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=(3, 4, 4),
                     stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2),

            # (64, seq_len//4, 8, 8) -> (128, 1, 4, 4)
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=(3, 4, 4),
                     stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2),

            # Global pooling
            nn.AdaptiveAvgPool3d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 4, base_filters),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(base_filters, 1)
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            sequence: (B, seq_len, C, H, W)

        Returns:
            score: (B, 1)
        """
        B, T, C, H, W = sequence.shape

        # Rearrange to (B, C, T, H, W) for 3D conv
        x = sequence.permute(0, 2, 1, 3, 4)

        # 3D convolutions
        features = self.conv3d(x)  # (B, base_filters*4, 1, 1, 1)
        features = features.view(B, -1)  # (B, base_filters*4)

        # Classify
        score = self.classifier(features)  # (B, 1)

        return score


def test_sequence_critic():
    """Test sequence critic"""
    batch_size = 8
    seq_len = 5
    in_channels = 8
    img_size = 32

    critic = SequenceCritic(
        in_channels=in_channels,
        seq_len=seq_len,
        cnn_embed_dim=256,
        lstm_hidden=128,
        lstm_layers=2
    )

    sequence = torch.randn(batch_size, seq_len, in_channels, img_size, img_size)
    score = critic(sequence)

    print(f"Sequence Critic test:")
    print(f"  Input sequence: {sequence.shape}")
    print(f"  Output score: {score.shape}")
    print(f"  Parameters: {sum(p.numel() for p in critic.parameters()):,}")

    # Test temporal features
    temporal_features = critic.get_temporal_features(sequence)
    print(f"  Temporal features: {temporal_features.shape}")

    assert score.shape == (batch_size, 1)
    print("  ✓ Sequence critic test passed!")


def test_simplified_critic():
    """Test simplified 3D conv critic"""
    batch_size = 8
    seq_len = 5
    in_channels = 8
    img_size = 32

    critic = SimplifiedSequenceCritic(in_channels=in_channels, seq_len=seq_len)

    sequence = torch.randn(batch_size, seq_len, in_channels, img_size, img_size)
    score = critic(sequence)

    print(f"\nSimplified Sequence Critic (3D Conv) test:")
    print(f"  Input sequence: {sequence.shape}")
    print(f"  Output score: {score.shape}")
    print(f"  Parameters: {sum(p.numel() for p in critic.parameters()):,}")

    assert score.shape == (batch_size, 1)
    print("  ✓ Simplified critic test passed!")


if __name__ == '__main__':
    test_sequence_critic()
    test_simplified_critic()
