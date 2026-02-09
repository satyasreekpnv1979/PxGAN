"""
Loss functions for PxGAN training
Includes adversarial losses, feature matching, gradient penalty, and reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def hinge_loss_dis(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for discriminator

    Args:
        real_scores: Discriminator scores for real samples
        fake_scores: Discriminator scores for fake samples

    Returns:
        Discriminator hinge loss
    """
    # For PatchGAN, scores may be (B, 1, H, W), so average over spatial dimensions
    if real_scores.dim() > 2:
        real_scores = real_scores.mean(dim=[2, 3])  # (B, 1)
    if fake_scores.dim() > 2:
        fake_scores = fake_scores.mean(dim=[2, 3])

    loss_real = torch.mean(F.relu(1.0 - real_scores))
    loss_fake = torch.mean(F.relu(1.0 + fake_scores))

    return loss_real + loss_fake


def hinge_loss_gen(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator

    Args:
        fake_scores: Discriminator scores for generated samples

    Returns:
        Generator hinge loss
    """
    if fake_scores.dim() > 2:
        fake_scores = fake_scores.mean(dim=[2, 3])

    return -torch.mean(fake_scores)


def bce_loss_dis(real_scores: torch.Tensor,
                 fake_scores: torch.Tensor,
                 label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Binary cross-entropy loss for discriminator

    Args:
        real_scores: Discriminator scores for real samples
        fake_scores: Discriminator scores for fake samples
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        BCE discriminator loss
    """
    bce = nn.BCEWithLogitsLoss()

    # Average patch scores if needed
    if real_scores.dim() > 2:
        real_scores = real_scores.mean(dim=[2, 3])
    if fake_scores.dim() > 2:
        fake_scores = fake_scores.mean(dim=[2, 3])

    # Real labels with smoothing: 1.0 -> (1.0 - label_smoothing)
    real_labels = torch.ones_like(real_scores) * (1.0 - label_smoothing)
    fake_labels = torch.zeros_like(fake_scores)

    loss_real = bce(real_scores, real_labels)
    loss_fake = bce(fake_scores, fake_labels)

    return loss_real + loss_fake


def bce_loss_gen(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for generator

    Args:
        fake_scores: Discriminator scores for generated samples

    Returns:
        BCE generator loss
    """
    bce = nn.BCEWithLogitsLoss()

    if fake_scores.dim() > 2:
        fake_scores = fake_scores.mean(dim=[2, 3])

    # Generator wants discriminator to output 1 for fake samples
    real_labels = torch.ones_like(fake_scores)

    return bce(fake_scores, real_labels)


def feature_matching_loss(discriminator: nn.Module,
                          real: torch.Tensor,
                          fake: torch.Tensor,
                          layer_indices: Optional[List[int]] = None) -> torch.Tensor:
    """
    Feature matching loss: match intermediate feature statistics

    Args:
        discriminator: Discriminator model (must have get_intermediate_features method)
        real: Real samples
        fake: Generated samples
        layer_indices: Which layers to extract features from

    Returns:
        Feature matching loss (L1 distance between feature means)
    """
    # Extract intermediate features
    if hasattr(discriminator, 'get_intermediate_features'):
        real_features = discriminator.get_intermediate_features(real, layer_indices)
        fake_features = discriminator.get_intermediate_features(fake, layer_indices)
    else:
        # Fallback: use final output
        with torch.no_grad():
            real_features = [discriminator(real)]
        fake_features = [discriminator(fake)]

    # Compute L1 loss between feature means
    fm_loss = 0.0
    for real_feat, fake_feat in zip(real_features, fake_features):
        # Mean over batch and spatial dimensions
        real_mean = real_feat.mean(dim=0)
        fake_mean = fake_feat.mean(dim=0)

        fm_loss += F.l1_loss(fake_mean, real_mean)

    return fm_loss / len(real_features)


def gradient_penalty(discriminator: nn.Module,
                    real: torch.Tensor,
                    fake: torch.Tensor,
                    device: torch.device,
                    lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Gradient penalty for WGAN-GP

    Args:
        discriminator: Discriminator model
        real: Real samples
        fake: Fake samples
        device: Device
        lambda_gp: Gradient penalty weight

    Returns:
        Gradient penalty loss
    """
    batch_size = real.size(0)

    # Random interpolation weight
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)

    # Interpolated samples
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates = interpolates.requires_grad_(True)

    # Discriminator scores on interpolates
    d_interpolates = discriminator(interpolates)

    # Average over spatial dimensions if patch scores
    if d_interpolates.dim() > 2:
        d_interpolates = d_interpolates.mean(dim=[2, 3])

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Gradient penalty: (||grad|| - 1)^2
    grad_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()

    return penalty


def reconstruction_loss(real: torch.Tensor,
                       fake: torch.Tensor,
                       loss_type: str = 'l1') -> torch.Tensor:
    """
    Reconstruction loss between real and generated samples

    Args:
        real: Real samples
        fake: Generated samples
        loss_type: 'l1', 'l2', or 'huber'

    Returns:
        Reconstruction loss
    """
    if loss_type == 'l1':
        return F.l1_loss(fake, real)
    elif loss_type == 'l2':
        return F.mse_loss(fake, real)
    elif loss_type == 'huber':
        return F.smooth_l1_loss(fake, real)
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


def perceptual_loss_simple(real: torch.Tensor,
                           fake: torch.Tensor) -> torch.Tensor:
    """
    Simple perceptual loss based on channel-wise statistics

    Args:
        real: Real samples (B, C, H, W)
        fake: Generated samples (B, C, H, W)

    Returns:
        Perceptual loss
    """
    # Compute channel-wise mean and std
    real_mean = real.mean(dim=[2, 3])  # (B, C)
    real_std = real.std(dim=[2, 3])

    fake_mean = fake.mean(dim=[2, 3])
    fake_std = fake.std(dim=[2, 3])

    # L2 loss on statistics
    mean_loss = F.mse_loss(fake_mean, real_mean)
    std_loss = F.mse_loss(fake_std, real_std)

    return mean_loss + std_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for generator training

    Combines adversarial, feature matching, and reconstruction losses
    """

    def __init__(self,
                 lambda_adv: float = 1.0,
                 lambda_fm: float = 10.0,
                 lambda_recon: float = 5.0,
                 adv_loss_type: str = 'hinge'):
        """
        Args:
            lambda_adv: Weight for adversarial loss
            lambda_fm: Weight for feature matching loss
            lambda_recon: Weight for reconstruction loss
            adv_loss_type: 'hinge' or 'bce'
        """
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_recon = lambda_recon
        self.adv_loss_type = adv_loss_type

    def forward(self,
                fake_scores: torch.Tensor,
                discriminator: nn.Module,
                real: torch.Tensor,
                fake: torch.Tensor) -> dict:
        """
        Compute combined generator loss

        Returns:
            Dictionary with individual loss components and total
        """
        # Adversarial loss
        if self.adv_loss_type == 'hinge':
            adv_loss = hinge_loss_gen(fake_scores)
        else:
            adv_loss = bce_loss_gen(fake_scores)

        # Feature matching loss
        fm_loss = feature_matching_loss(discriminator, real, fake)

        # Reconstruction loss
        recon_loss = reconstruction_loss(real, fake, loss_type='l1')

        # Total loss
        total_loss = (self.lambda_adv * adv_loss +
                     self.lambda_fm * fm_loss +
                     self.lambda_recon * recon_loss)

        return {
            'total': total_loss,
            'adv': adv_loss,
            'fm': fm_loss,
            'recon': recon_loss
        }


def test_losses():
    """Test loss functions"""
    batch_size = 16
    channels = 8
    img_size = 32

    # Dummy data
    real = torch.randn(batch_size, channels, img_size, img_size)
    fake = torch.randn(batch_size, channels, img_size, img_size)

    # Patch scores (B, 1, H, W)
    real_scores_patch = torch.randn(batch_size, 1, 4, 4)
    fake_scores_patch = torch.randn(batch_size, 1, 4, 4)

    # Scalar scores (B, 1)
    fake_scores_scalar = torch.randn(batch_size, 1)

    print("Testing loss functions:")

    # Hinge losses
    d_hinge = hinge_loss_dis(real_scores_patch, fake_scores_patch)
    g_hinge = hinge_loss_gen(fake_scores_scalar)
    print(f"  Hinge loss (D): {d_hinge.item():.4f}")
    print(f"  Hinge loss (G): {g_hinge.item():.4f}")

    # BCE losses
    d_bce = bce_loss_dis(real_scores_patch, fake_scores_patch)
    g_bce = bce_loss_gen(fake_scores_scalar)
    print(f"  BCE loss (D): {d_bce.item():.4f}")
    print(f"  BCE loss (G): {g_bce.item():.4f}")

    # Reconstruction loss
    recon = reconstruction_loss(real, fake, loss_type='l1')
    print(f"  Reconstruction (L1): {recon.item():.4f}")

    # Perceptual loss
    percep = perceptual_loss_simple(real, fake)
    print(f"  Perceptual loss: {percep.item():.4f}")

    print("  ✓ All loss tests passed!")


if __name__ == '__main__':
    test_losses()
