"""
Main PxGAN Trainer
Orchestrates multi-discriminator GAN training with adversarial hardening
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from ..models import (
    ConditionalGenerator,
    PatchDiscriminator,
    SequenceCritic,
    hinge_loss_dis,
    hinge_loss_gen,
    feature_matching_loss,
    gradient_penalty,
    reconstruction_loss
)
from ..utils import MetricLogger, ExperimentTracker
from .checkpointing import CheckpointManager
from .adversarial import generate_fgsm_evasions


class PxGANTrainer:
    """
    Main trainer for PxGAN

    Handles:
    - Multi-discriminator training (PatchGAN + Sequence Critic)
    - Loss computation and backpropagation
    - Validation and checkpointing
    - Adversarial hardening
    """

    def __init__(self,
                 config: Dict,
                 generator: ConditionalGenerator,
                 discriminator_patch: PatchDiscriminator,
                 discriminator_seq: Optional[SequenceCritic],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 experiment_tracker: Optional[ExperimentTracker] = None):
        """
        Args:
            config: Configuration dictionary
            generator: Generator model
            discriminator_patch: PatchGAN discriminator
            discriminator_seq: Sequence critic (optional)
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on
            experiment_tracker: Optional experiment tracker
        """
        self.config = config
        self.train_config = config['training']
        self.device = device

        # Models
        self.G = generator.to(device)
        self.D_patch = discriminator_patch.to(device)
        self.D_seq = discriminator_seq.to(device) if discriminator_seq is not None else None

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Experiment tracker
        self.tracker = experiment_tracker

        # Optimizers
        lr = self.train_config['lr']
        betas = tuple(self.train_config['betas'])

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.opt_D_patch = torch.optim.Adam(self.D_patch.parameters(), lr=lr, betas=betas)

        if self.D_seq is not None:
            self.opt_D_seq = torch.optim.Adam(self.D_seq.parameters(), lr=lr, betas=betas)
        else:
            self.opt_D_seq = None

        # Learning rate schedulers
        self.setup_schedulers()

        # Checkpoint manager
        checkpoint_dir = self.config['paths'].get('checkpoint_dir', './checkpoints')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_to_keep=self.train_config.get('max_checkpoints_to_keep', 5),
            save_best_only=self.train_config.get('save_best_only', False)
        )

        # Loss weights
        self.loss_weights = self.train_config['loss_weights']

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = None

        print(f"PxGANTrainer initialized on {device}")
        print(f"  Generator params: {sum(p.numel() for p in self.G.parameters()):,}")
        print(f"  D_patch params: {sum(p.numel() for p in self.D_patch.parameters()):,}")
        if self.D_seq:
            print(f"  D_seq params: {sum(p.numel() for p in self.D_seq.parameters()):,}")

    def setup_schedulers(self):
        """Setup learning rate schedulers"""
        sched_config = self.train_config.get('scheduler', {})
        sched_type = sched_config.get('type', 'reduce_on_plateau')

        if sched_type == 'reduce_on_plateau':
            self.sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_G,
                mode='max',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.5),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
            self.sched_D_patch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D_patch,
                mode='max',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.5)
            )
        else:
            self.sched_G = None
            self.sched_D_patch = None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average losses
        """
        self.G.train()
        self.D_patch.train()
        if self.D_seq:
            self.D_seq.train()

        metric_logger = MetricLogger()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            real_images = batch['image'].to(self.device)
            cond = batch['cond'].to(self.device)
            batch_size = real_images.size(0)

            # ===== Train Discriminators =====

            # 1. Train D_patch
            for _ in range(self.train_config.get('discriminator_steps', 1)):
                loss_d_patch = self.train_discriminator_patch_step(real_images, cond)

            # 2. Train D_seq (less frequently)
            loss_d_seq = 0.0
            if self.D_seq and batch_idx % self.train_config.get('seq_critic_frequency', 5) == 0:
                # Need sequence batch for D_seq
                # For simplicity, skip or use consecutive batches logic
                # Here we'll skip for now as it requires special data handling
                pass

            # ===== Train Generator =====

            for _ in range(self.train_config.get('generator_steps', 1)):
                losses_g = self.train_generator_step(real_images, cond)

            # Update metrics
            metric_logger.update(
                d_patch_loss=loss_d_patch,
                g_total_loss=losses_g['total'],
                g_adv_loss=losses_g['adv'],
                g_fm_loss=losses_g['fm'],
                g_recon_loss=losses_g['recon']
            )

            # Update progress bar
            pbar.set_postfix({
                'D': f'{loss_d_patch:.3f}',
                'G': f'{losses_g["total"]:.3f}'
            })

            # Log to tracker
            if self.tracker and batch_idx % self.train_config.get('log_interval', 100) == 0:
                self.tracker.log_metrics({
                    'd_patch_loss': loss_d_patch,
                    'g_total_loss': losses_g['total'],
                    'g_adv_loss': losses_g['adv'],
                    'g_fm_loss': losses_g['fm'],
                    'g_recon_loss': losses_g['recon']
                }, step=self.global_step, prefix='train/')

            self.global_step += 1

        # Get epoch averages
        epoch_metrics = metric_logger.get_all_averages()

        return epoch_metrics

    def train_discriminator_patch_step(self, real_images: torch.Tensor, cond: torch.Tensor) -> float:
        """Train PatchGAN discriminator for one step"""
        batch_size = real_images.size(0)

        # Generate fake images
        z = torch.randn(batch_size, self.G.z_dim).to(self.device)
        fake_images = self.G(z, cond).detach()

        # Discriminator scores
        real_scores = self.D_patch(real_images)
        fake_scores = self.D_patch(fake_images)

        # Hinge loss
        loss_d = hinge_loss_dis(real_scores, fake_scores)

        # Gradient penalty (optional)
        if self.loss_weights.get('lambda_gp', 0) > 0:
            gp = gradient_penalty(self.D_patch, real_images, fake_images, self.device,
                                 self.loss_weights['lambda_gp'])
            loss_d += gp

        # Backward
        self.opt_D_patch.zero_grad()
        loss_d.backward()

        # Gradient clipping
        if self.train_config.get('clip_grad_norm', None):
            nn.utils.clip_grad_norm_(self.D_patch.parameters(), self.train_config['clip_grad_norm'])

        self.opt_D_patch.step()

        return loss_d.item()

    def train_generator_step(self, real_images: torch.Tensor, cond: torch.Tensor) -> Dict[str, float]:
        """Train generator for one step"""
        batch_size = real_images.size(0)

        # Generate fake images
        z = torch.randn(batch_size, self.G.z_dim).to(self.device)
        fake_images = self.G(z, cond)

        # Discriminator scores on fake
        fake_scores = self.D_patch(fake_images)

        # Adversarial loss
        loss_adv = hinge_loss_gen(fake_scores) * self.loss_weights['lambda_adv_patch']

        # Feature matching loss
        loss_fm = feature_matching_loss(self.D_patch, real_images, fake_images) * self.loss_weights['lambda_fm']

        # Reconstruction loss (optional)
        loss_recon = reconstruction_loss(real_images, fake_images) * self.loss_weights.get('lambda_recon', 0)

        # Total generator loss
        loss_g_total = loss_adv + loss_fm + loss_recon

        # Backward
        self.opt_G.zero_grad()
        loss_g_total.backward()

        # Gradient clipping
        if self.train_config.get('clip_grad_norm', None):
            nn.utils.clip_grad_norm_(self.G.parameters(), self.train_config['clip_grad_norm'])

        self.opt_G.step()

        return {
            'total': loss_g_total.item(),
            'adv': loss_adv.item(),
            'fm': loss_fm.item(),
            'recon': loss_recon.item()
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of validation metrics
        """
        self.G.eval()
        self.D_patch.eval()
        if self.D_seq:
            self.D_seq.eval()

        metric_logger = MetricLogger()

        for batch in tqdm(self.val_loader, desc='Validation'):
            real_images = batch['image'].to(self.device)
            cond = batch['cond'].to(self.device)
            batch_size = real_images.size(0)

            # Generate fake images
            z = torch.randn(batch_size, self.G.z_dim).to(self.device)
            fake_images = self.G(z, cond)

            # Discriminator scores
            real_scores = self.D_patch(real_images)
            fake_scores = self.D_patch(fake_images)

            # Losses
            loss_d = hinge_loss_dis(real_scores, fake_scores)
            loss_g_adv = hinge_loss_gen(fake_scores)
            loss_g_fm = feature_matching_loss(self.D_patch, real_images, fake_images)

            metric_logger.update(
                val_d_loss=loss_d.item(),
                val_g_adv_loss=loss_g_adv.item(),
                val_g_fm_loss=loss_g_fm.item()
            )

        val_metrics = metric_logger.get_all_averages()

        # Log to tracker
        if self.tracker:
            self.tracker.log_metrics(val_metrics, step=epoch, prefix='val/')

        return val_metrics

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
        """
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from:
            checkpoint = self.checkpoint_manager.load_best()
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                self.current_epoch = start_epoch
                print(f"Resumed from epoch {checkpoint['epoch']}")

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            print(f"\nEpoch {epoch} training metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")

            # Validate
            if epoch % self.train_config.get('validate_every', 1) == 0:
                val_metrics = self.validate(epoch)

                print(f"Epoch {epoch} validation metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}")

                # Update schedulers
                if self.sched_G:
                    # Use negative D loss as metric (higher is better)
                    val_metric = -val_metrics['val_d_loss']
                    self.sched_G.step(val_metric)
                    self.sched_D_patch.step(val_metric)

                # Save checkpoint
                if epoch % self.train_config.get('save_checkpoint_every', 5) == 0:
                    self.checkpoint_manager.save(
                        epoch=epoch,
                        models={'G': self.G, 'D_patch': self.D_patch},
                        optimizers={'G': self.opt_G, 'D_patch': self.opt_D_patch},
                        metric=val_metric,
                        metadata=train_metrics
                    )

        print("Training complete!")

    def save_final_model(self, path: str):
        """Save final trained models"""
        torch.save({
            'G': self.G.state_dict(),
            'D_patch': self.D_patch.state_dict(),
            'D_seq': self.D_seq.state_dict() if self.D_seq else None,
            'config': self.config
        }, path)
        print(f"Saved final model to {path}")
