"""
Adversarial training and evasion generation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def generate_fgsm_evasions(model: nn.Module,
                           images: torch.Tensor,
                           labels: torch.Tensor,
                           epsilon: float = 0.01,
                           targeted: bool = False) -> torch.Tensor:
    """
    Generate FGSM (Fast Gradient Sign Method) adversarial examples

    Args:
        model: Target model (discriminator)
        images: Input images
        labels: Target labels (0 for normal, 1 for fake)
        epsilon: Perturbation magnitude
        targeted: Whether to perform targeted attack

    Returns:
        Adversarial examples
    """
    images = images.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images)

    # Average patch scores if needed
    if outputs.dim() > 2:
        outputs = outputs.mean(dim=[2, 3])

    # Loss
    loss = F.binary_cross_entropy_with_logits(
        outputs,
        labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
    )

    # Backward
    model.zero_grad()
    loss.backward()

    # Get gradients
    data_grad = images.grad.data

    # Create perturbation
    if targeted:
        perturbation = -epsilon * data_grad.sign()
    else:
        perturbation = epsilon * data_grad.sign()

    # Apply perturbation
    perturbed_images = images + perturbation

    # Clamp to valid range (assuming images in [-1, 1] from Tanh)
    perturbed_images = torch.clamp(perturbed_images, -1.0, 1.0)

    return perturbed_images.detach()


def generate_pgd_evasions(model: nn.Module,
                         images: torch.Tensor,
                         labels: torch.Tensor,
                         epsilon: float = 0.01,
                         alpha: float = 0.002,
                         num_steps: int = 10) -> torch.Tensor:
    """
    Generate PGD (Projected Gradient Descent) adversarial examples

    Args:
        model: Target model
        images: Input images
        labels: Target labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_steps: Number of PGD steps

    Returns:
        Adversarial examples
    """
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()

    for _ in range(num_steps):
        perturbed_images.requires_grad = True

        # Forward pass
        outputs = model(perturbed_images)

        if outputs.dim() > 2:
            outputs = outputs.mean(dim=[2, 3])

        # Loss
        loss = F.binary_cross_entropy_with_logits(
            outputs,
            labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
        )

        # Backward
        model.zero_grad()
        loss.backward()

        # Update perturbation
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach() + alpha * data_grad.sign()

        # Project back to epsilon ball
        perturbation = torch.clamp(perturbed_images - original_images, -epsilon, epsilon)
        perturbed_images = original_images + perturbation

        # Clamp to valid range
        perturbed_images = torch.clamp(perturbed_images, -1.0, 1.0)

    return perturbed_images.detach()


def generate_gan_evasions(generator: nn.Module,
                         discriminator: nn.Module,
                         cond: torch.Tensor,
                         device: torch.device,
                         z_search_steps: int = 10,
                         lr: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate evasive examples by optimizing latent code z

    Goal: find z such that G(z) fools discriminator

    Args:
        generator: Generator model
        discriminator: Discriminator model
        cond: Conditional features
        device: Device
        z_search_steps: Number of optimization steps
        lr: Learning-rate for z optimization

    Returns:
        (evasive_images, optimized_z)
    """
    batch_size = cond.size(0)
    z_dim = generator.z_dim

    # Initialize z
    z = torch.randn(batch_size, z_dim, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(z_search_steps):
        optimizer.zero_grad()

        # Generate images
        fake_images = generator(z, cond)

        # Discriminator scores
        d_scores = discriminator(fake_images)

        if d_scores.dim() > 2:
            d_scores = d_scores.mean(dim=[2, 3])

        # Maximize discriminator score (fool it to think images are real)
        # Equivalently, minimize -score
        loss = -d_scores.mean()

        loss.backward()
        optimizer.step()

    # Generate final evasive images
    with torch.no_grad():
        evasive_images = generator(z, cond)

    return evasive_images.detach(), z.detach()


def adversarial_training_step(discriminator: nn.Module,
                              optimizer: torch.optim.Optimizer,
                              real_images: torch.Tensor,
                              evasive_images: torch.Tensor,
                              loss_fn) -> float:
    """
    Adversarial training step: fine-tune discriminator on evasive examples

    Args:
        discriminator: Discriminator model
        optimizer: Discriminator optimizer
        real_images: Real images
        evasive_images: Adversarial/evasive images
        loss_fn: Loss function

    Returns:
        Loss value
    """
    optimizer.zero_grad()

    # Discriminator scores
    real_scores = discriminator(real_images)
    fake_scores = discriminator(evasive_images)

    # Loss
    loss = loss_fn(real_scores, fake_scores)

    loss.backward()
    optimizer.step()

    return loss.item()


def compute_adversarial_robustness(model: nn.Module,
                                   dataloader,
                                   epsilon: float = 0.01,
                                   device: torch.device = None) -> dict:
    """
    Compute adversarial robustness metrics

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        epsilon: Attack epsilon
        device: Device

    Returns:
        Dictionary with robustness metrics
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()

    total = 0
    correct_clean = 0
    correct_adv = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(images)

            if outputs_clean.dim() > 2:
                outputs_clean = outputs_clean.mean(dim=[2, 3])

            preds_clean = (torch.sigmoid(outputs_clean) > 0.5).long().squeeze()
            correct_clean += (preds_clean == labels).sum().item()

        # Adversarial examples
        adv_images = generate_fgsm_evasions(model, images, labels, epsilon=epsilon)

        # Adversarial accuracy
        with torch.no_grad():
            outputs_adv = model(adv_images)

            if outputs_adv.dim() > 2:
                outputs_adv = outputs_adv.mean(dim=[2, 3])

            preds_adv = (torch.sigmoid(outputs_adv) > 0.5).long().squeeze()
            correct_adv += (preds_adv == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * correct_clean / total
    adv_acc = 100.0 * correct_adv / total
    robustness = adv_acc / clean_acc if clean_acc > 0 else 0.0

    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'robustness_ratio': robustness,
        'attack_success_rate': 100.0 - adv_acc
    }
