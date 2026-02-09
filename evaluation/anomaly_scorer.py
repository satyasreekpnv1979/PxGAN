"""
Basic anomaly scorer (simplified for quick implementation)
Can be extended with full reconstruction + statistical scoring
"""

import torch
import numpy as np
from typing import Optional


class AnomalyScorer:
    """Simple anomaly scorer using discriminator scores"""

    def __init__(self, discriminator, mapper, weights=None):
        self.discriminator = discriminator
        self.mapper = mapper
        self.weights = weights or [1.0]

    @torch.no_grad()
    def score(self, window_df, device='cpu') -> float:
        """Score a single window"""
        # Convert to image
        image = self.mapper.transform(window_df)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

        # Discriminator score (lower = more anomalous)
        self.discriminator.eval()
        d_score = self.discriminator(image_tensor)

        if d_score.dim() > 2:
            d_score = d_score.mean()

        # Return negative score (higher = more anomalous)
        return -d_score.item()

    @torch.no_grad()
    def score_batch(self, windows, device='cpu') -> np.ndarray:
        """Score a batch of windows"""
        scores = []
        for window in windows:
            score = self.score(window, device)
            scores.append(score)
        return np.array(scores)
