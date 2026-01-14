"""
Training script for OCTCube MAE.

Can be used to:
1. Fine-tune pretrained OCTCube MAE on your dataset
2. Train MAE from scratch

For anomaly detection, we train the MAE to reconstruct patches.
Patches that are hard to reconstruct after training are likely anomalous.
"""

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional
import time
import json

from spectralis_dataset import SpectralisLoader
from octcube import OCTCubeMAE


@dataclass
class MAETrainConfig:
    # Model parameters
    img_size: int = 512
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'

    # Training parameters
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 1  # Per-volume, processed in chunks
    mask_ratio: float = 0.0  # 0.0 = reconstruct all patches (autoencoder), 0.75 = standard MAE

    # Checkpointing
    save_dir: str = "checkpoints_mae"
    save_interval: int = 5  # Save every N epochs
    log_interval: int = 10  # Log every N steps

    # Paths
    pretrained_checkpoint: Optional[str] = None  # Start from pretrained OCTCube
    resume_from: Optional[str] = None  # Resume training


class MAETrainer:
    def __init__(self, config: MAETrainConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.global_step = 0

    def create_model(self) -> OCTCubeMAE:
        """Create MAE model."""
        model = OCTCubeMAE(
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            num_frames=self.config.num_frames,
            t_patch_size=self.config.t_patch_size,
            size=self.config.model_size,
            checkpoint_path=self.config.pretrained_checkpoint,
        )
        return model.to(self.device)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Random masking for MAE training.

        Args:
            x: (B, T, L, D) tokens
            mask_ratio: fraction of tokens to mask

        Returns:
            x_masked: tokens with masked positions removed
            mask: (B, T*L) binary mask (1 = masked/removed)
            ids_restore: indices to restore original order
        """
        B, T, L, D = x.shape
        N = T * L
        len_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]

        # Flatten x for gathering
        x_flat = x.reshape(B, N, D)
        x_masked = torch.gather(x_flat, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 1 = masked, 0 = keep
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def compute_loss(
        self,
        model: OCTCubeMAE,
        images: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> tuple:
        """
        Compute reconstruction loss.

        Args:
            model: MAE model
            images: (B, T, C, H, W) input
            mask_ratio: if > 0, only compute loss on masked patches

        Returns:
            loss: scalar loss
            metrics: dict with additional metrics
        """
        B, T, C, H, W = images.shape

        # Get reconstruction and tokens
        reconstructed_patches, tokens = model(images)

        # Get target patches
        target_patches = model.patchify(images)

        if mask_ratio > 0:
            # MAE-style: only compute loss on masked patches
            T_p, L_p = tokens.shape[1], tokens.shape[2]
            N = T_p * L_p

            # Create random mask
            noise = torch.rand(B, N, device=images.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            len_keep = int(N * (1 - mask_ratio))

            # Mask: 1 = compute loss, 0 = ignore
            mask = torch.ones(B, N, device=images.device)
            mask.scatter_(1, ids_shuffle[:, :len_keep], 0)
            mask = mask.reshape(B, T_p, L_p)

            # Compute loss only on masked patches
            diff = (reconstructed_patches - target_patches) ** 2
            diff = diff.mean(dim=-1)  # (B, T_p, L_p)

            loss = (diff * mask).sum() / mask.sum()
            mask_pct = mask.mean().item()
        else:
            # Autoencoder-style: compute loss on all patches
            loss = F.mse_loss(reconstructed_patches, target_patches)
            mask_pct = 1.0

        # Per-patch losses for analysis
        with torch.no_grad():
            patch_losses = ((reconstructed_patches - target_patches) ** 2).mean(dim=-1)
            metrics = {
                "loss": loss.item(),
                "loss_mean": patch_losses.mean().item(),
                "loss_max": patch_losses.max().item(),
                "loss_std": patch_losses.std().item(),
                "mask_pct": mask_pct,
            }

        return loss, metrics

    @torch.no_grad()
    def validate(self, model: OCTCubeMAE, val_loader) -> dict:
        """Run validation."""
        model.eval()
        total_loss = 0
        total_samples = 0

        for batch in val_loader:
            images = batch["frames"].permute(1, 0, 2, 3).to(self.device)
            if images.shape[1] != 1:
                images = images[:, :1]

            S = images.shape[0]

            # Process in chunks
            for start in range(0, S, self.config.num_frames):
                end = min(start + self.config.num_frames, S)
                chunk = images[start:end]

                # Pad if needed
                if chunk.shape[0] % self.config.t_patch_size != 0:
                    pad = self.config.t_patch_size - (chunk.shape[0] % self.config.t_patch_size)
                    chunk = torch.cat([chunk, chunk[-1:].expand(pad, -1, -1, -1)], dim=0)

                chunk = chunk.unsqueeze(0)  # (1, T, C, H, W)
                loss, _ = self.compute_loss(model, chunk, mask_ratio=0.0)
                total_loss += loss.item()
                total_samples += 1

        model.train()
        return {"val_loss": total_loss / max(total_samples, 1)}

    def train_epoch(
        self,
        model: OCTCubeMAE,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        train_loader,
    ) -> list:
        """Train for one epoch."""
        model.train()
        epoch_metrics = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["frames"].permute(1, 0, 2, 3).to(self.device)
            if images.shape[1] != 1:
                images = images[:, :1]

            S = images.shape[0]

            # Process volume in chunks
            for start in range(0, S, self.config.num_frames):
                end = min(start + self.config.num_frames, S)
                chunk = images[start:end]

                # Pad if needed
                if chunk.shape[0] % self.config.t_patch_size != 0:
                    pad = self.config.t_patch_size - (chunk.shape[0] % self.config.t_patch_size)
                    chunk = torch.cat([chunk, chunk[-1:].expand(pad, -1, -1, -1)], dim=0)

                chunk = chunk.unsqueeze(0)  # (1, T, C, H, W)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    loss, metrics = self.compute_loss(
                        model, chunk,
                        mask_ratio=self.config.mask_ratio
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_metrics.append(metrics)
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    print(f"  Step {self.global_step}: loss={metrics['loss']:.4f}")

        return epoch_metrics

    def save_checkpoint(self, model, optimizer, scaler, path: str):
        """Save training checkpoint."""
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config.__dict__,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, model, optimizer, scaler, path: str) -> bool:
        """Load training checkpoint."""
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
        return True

    def plot_losses(self, save_path: str):
        """Plot training curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if self.train_losses:
            ax.plot(self.train_losses, label="Train", alpha=0.7)
        if self.val_losses:
            epochs = np.arange(1, len(self.val_losses) + 1)
            ax.scatter(epochs * (len(self.train_losses) // len(self.val_losses)),
                      self.val_losses, label="Val", s=50, zorder=5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("MAE Training Loss")
        ax.legend()
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def train(self):
        """Main training loop."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Create model
        print("Creating model...")
        model = self.create_model()
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scaler = GradScaler("cuda")

        # Resume if specified
        if self.config.resume_from:
            self.load_checkpoint(model, optimizer, scaler, self.config.resume_from)

        # Create data loaders
        print("Loading data...")
        train_loader = torch.utils.data.DataLoader(
            SpectralisLoader(
                split_label="train",
                target_size=(self.config.img_size, self.config.img_size),
                normalize=True,
            ),
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )
        val_loader = torch.utils.data.DataLoader(
            SpectralisLoader(
                split_label="val",
                target_size=(self.config.img_size, self.config.img_size),
                normalize=True,
            ),
            batch_size=1,
            num_workers=4,
        )

        print(f"\nStarting training from epoch {self.current_epoch + 1}")
        print(f"Mask ratio: {self.config.mask_ratio}")

        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            start_time = time.time()

            # Train
            epoch_metrics = self.train_epoch(model, optimizer, scaler, train_loader)
            epoch_loss = np.mean([m["loss"] for m in epoch_metrics])
            self.train_losses.extend([m["loss"] for m in epoch_metrics])

            # Validate
            val_metrics = self.validate(model, val_loader)
            self.val_losses.append(val_metrics["val_loss"])

            epoch_time = time.time() - start_time
            print(f"  Train loss: {epoch_loss:.4f}, Val loss: {val_metrics['val_loss']:.4f}, Time: {epoch_time:.1f}s")

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    model, optimizer, scaler,
                    os.path.join(self.config.save_dir, f"epoch_{epoch}.pt")
                )

            # Always save latest
            self.save_checkpoint(
                model, optimizer, scaler,
                os.path.join(self.config.save_dir, "latest.pt")
            )

            # Plot
            self.plot_losses(os.path.join(self.config.save_dir, "loss_curves.png"))

        # Save final
        self.save_checkpoint(
            model, optimizer, scaler,
            os.path.join(self.config.save_dir, "final.pt")
        )

        print("\nTraining complete!")


def main():
    config = MAETrainConfig(
        pretrained_checkpoint="path/to/octcube_checkpoint.pth",  # UPDATE THIS
        save_dir="checkpoints_mae",
        epochs=50,
        lr=1e-4,
        mask_ratio=0.0,  # 0 = reconstruct all (good for anomaly detection)
    )

    trainer = MAETrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
