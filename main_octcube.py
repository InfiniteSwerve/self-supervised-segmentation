"""
Training script for OCTCube-based segmentation.

This script adapts the MIRAGE training pipeline to work with OCTCube,
which processes 3D OCT volumes instead of individual slices.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import numpy as np
from spectralis_dataset import SpectralisLoader
import matplotlib.pyplot as plt
import time
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import os
import shutil

from octcube import OCTCubeSegmenter

typechecked = jaxtyped(typechecker=beartype)


@dataclass
class TrainConfig:
    # Training parameters
    step_size: int = 48  # Number of slices per volume chunk (must be divisible by t_patch_size=3)
    partial_val_interval: int = 500
    train_save_im: int = 300
    plot_losses: int = 300
    epochs: int = 20
    dice_calc_interval: int = 50

    # Model parameters
    img_size: int = 512
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'
    num_classes: int = 12

    # Paths
    checkpoint_path: Optional[str] = None  # Path to OCTCube pretrained encoder weights
    save_dir: str = "checkpoints_octcube"
    resume_from: Optional[str] = None  # Path to resume training from (latest checkpoint)


class Metrics:
    def __init__(self):
        from collections import defaultdict

        self.data = {}
        for split in ["train", "val", "val_partial", "test"]:
            self.data[split] = {"iterations": [], "metrics": defaultdict(list)}
        self.current_iter = 0
        self.current_epoch = 0

    def append(self, split, metrics):
        if split == "train":
            self.current_iter += 1

        self.data[split]["iterations"].append(self.current_iter)
        for k, v in metrics.items():
            self.data[split]["metrics"][k].append(v)
        self.print_latest(splits=split)

    def print_latest(self, splits=None):
        if splits is None:
            splits = ["train", "val", "val_partial", "test"]
        elif isinstance(splits, str):
            splits = [splits]

        for split in splits:
            if len(self.data[split]["iterations"]) == 0:
                continue

            latest_iter = self.data[split]["iterations"][-1]
            metrics_str = ", ".join(
                f"{k}={v[-1]:.4f}"
                for k, v in self.data[split]["metrics"].items()
                if len(v) > 0
            )
            print(f"{split} [{self.current_epoch}:{latest_iter}]: {metrics_str}")

    def plot_metrics(self):
        print("Plotting Metrics")
        all_metrics = set()
        for split in self.data.values():
            all_metrics.update(split["metrics"].keys())

        if len(all_metrics) == 0:
            return

        fig, axes = plt.subplots(len(all_metrics), 1, figsize=(10, 4 * len(all_metrics)))
        if len(all_metrics) == 1:
            axes = [axes]

        for ax, metric_name in zip(axes, sorted(all_metrics)):
            for split in ["train", "val", "val_partial", "test"]:
                iters = self.data[split]["iterations"]
                vals = self.data[split]["metrics"][metric_name]
                if len(vals) > 0:
                    if split == "train":
                        ax.plot(iters, vals, alpha=0.7, label=split)
                    else:
                        ax.scatter(iters, vals, label=split, s=50, zorder=5)
            ax.set_ylabel(metric_name)
            if "dice" not in metric_name.lower():
                ax.set_yscale("log")
            ax.legend()
        axes[-1].set_xlabel("iteration")
        plt.tight_layout()
        plt.savefig("octcube_metrics.png")
        plt.close()

    def save(self, path):
        import json
        save_data = {
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
            "data": {
                split: {
                    "iterations": self.data[split]["iterations"],
                    "metrics": dict(self.data[split]["metrics"]),
                }
                for split in self.data
            },
        }
        with open(path, "w") as f:
            json.dump(save_data, f)

    def load(self, path):
        import json
        from collections import defaultdict
        with open(path, "r") as f:
            save_data = json.load(f)
        self.current_iter = save_data["current_iter"]
        self.current_epoch = save_data.get("current_epoch", 0)
        for split in save_data["data"]:
            self.data[split]["iterations"] = save_data["data"][split]["iterations"]
            self.data[split]["metrics"] = defaultdict(list, save_data["data"][split]["metrics"])

    def should_save_train_images(self):
        return self.current_iter % TrainConfig.train_save_im == 0

    def should_run_validation_partial_epoch(self):
        return self.current_iter % TrainConfig.partial_val_interval == 0

    def should_plot_losses(self):
        return self.current_iter % TrainConfig.plot_losses == 0

    def should_calc_dice(self):
        return self.current_iter % TrainConfig.dice_calc_interval == 0

    def get_latest_val_dice(self) -> Optional[float]:
        """Get the most recent validation dice_mean score."""
        dice_vals = self.data["val"]["metrics"].get("dice_mean", [])
        if dice_vals:
            return dice_vals[-1]
        return None


def save_checkpoint(model, optimizer, scaler, metrics, path, is_best=False):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": {
            "current_iter": metrics.current_iter,
            "current_epoch": metrics.current_epoch,
            "data": {
                split: {
                    "iterations": metrics.data[split]["iterations"],
                    "metrics": dict(metrics.data[split]["metrics"]),
                }
                for split in metrics.data
            },
        },
        "is_best": is_best,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, metrics, path):
    """Load training checkpoint."""
    from collections import defaultdict

    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return False

    checkpoint = torch.load(path, map_location="cuda", weights_only=False)

    model.head.load_state_dict(checkpoint["head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore metrics
    metrics_data = checkpoint["metrics"]
    metrics.current_iter = metrics_data["current_iter"]
    metrics.current_epoch = metrics_data["current_epoch"]
    for split in metrics_data["data"]:
        metrics.data[split]["iterations"] = metrics_data["data"][split]["iterations"]
        metrics.data[split]["metrics"] = defaultdict(
            list, metrics_data["data"][split]["metrics"]
        )

    print(f"Loaded checkpoint from {path} (epoch {metrics.current_epoch}, iter {metrics.current_iter})")
    return True


@typechecked
def heightmap_to_volume(
    heightmap: Float[Tensor, "batch layers width"], imshape, one_hot=False
) -> Tensor:
    """
    Convert boundary heightmap to volume mask.

    heightmap: (B, K, W) tensor on GPU, may contain negative values for invalid
    Returns: (B, H, W) class indices, or (B, K+1, H, W) one-hot if one_hot=True
    """
    B, K, W = heightmap.shape
    boundaries = heightmap.clone()
    boundaries[boundaries < 0] = float("inf")

    y_coords = torch.arange(
        imshape[1], device=heightmap.device, dtype=heightmap.dtype
    ).view(1, 1, imshape[1], 1)
    boundaries = boundaries.unsqueeze(2)

    above_boundary = (y_coords >= boundaries).float()
    mask_volume = above_boundary.sum(dim=1).long()

    if one_hot:
        one_hot_volume = F.one_hot(mask_volume, num_classes=K + 1)
        return one_hot_volume.permute(0, 3, 1, 2).float()
    return mask_volume


@typechecked
def compute_dice_scores(
    gt_boundaries: Float[Tensor, "batch layers width"],
    pred_vol: Float[Tensor, "batch num_classes height width"],
    imshape: tuple,
) -> dict[str, float]:
    num_classes = pred_vol.shape[1]
    pred_vol = pred_vol.argmax(1)
    gt_vol = heightmap_to_volume(gt_boundaries, imshape)

    pred_oh = F.one_hot(pred_vol, num_classes).permute(0, 3, 1, 2).float()
    gt_oh = F.one_hot(gt_vol, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_oh * gt_oh).sum(dim=(0, 2, 3))
    union = pred_oh.sum(dim=(0, 2, 3)) + gt_oh.sum(dim=(0, 2, 3))

    dice = torch.where(union > 0, 2 * intersection / union, torch.ones_like(union))

    dice_scores = {f"dice_L{c}": dice[c].item() for c in range(num_classes)}
    dice_scores["dice_mean"] = dice.mean().item()
    return dice_scores


def supervised_epoch(
    model,
    optimizer,
    scaler,
    train_dataloader,
    val_dataloader,
    metrics: Metrics,
):
    """Training epoch - processes volumes in chunks."""
    model.train()

    for batch in train_dataloader:
        st = time.monotonic()

        # Load volume data
        # frames: (C, S, H, W) -> (S, C, H, W)
        ims = batch["frames"].permute(1, 0, 2, 3).cuda()
        # label: (C, L, S, W) -> (S, L, W) - index into C dim since C=1
        height = batch["label"][0].permute(1, 0, 2).cuda()

        S, C, H, W = ims.shape
        print(f"Volume loaded: {S} slices, {H}x{W}, time: {time.monotonic() - st:.2f}s")

        # Process in chunks of step_size slices
        for run in range(0, S, TrainConfig.step_size):
            end_idx = min(run + TrainConfig.step_size, S)
            chunk_size = end_idx - run

            # Need chunk_size divisible by t_patch_size for OCTCube
            # Pad if necessary
            if chunk_size % TrainConfig.t_patch_size != 0:
                pad_size = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                actual_end = min(end_idx + pad_size, S)
                if actual_end - run < chunk_size + pad_size:
                    # Pad by repeating last slice
                    chunk_ims = ims[run:end_idx]
                    chunk_height = height[run:end_idx]
                    pad_needed = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                    chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_needed, -1, -1, -1)], dim=0)
                    chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_needed, -1, -1)], dim=0)
                else:
                    chunk_ims = ims[run:actual_end]
                    chunk_height = height[run:actual_end]
            else:
                chunk_ims = ims[run:end_idx]
                chunk_height = height[run:end_idx]

            st = time.monotonic()
            metrics.append(
                "train",
                one_supervised_step(
                    model,
                    optimizer,
                    scaler,
                    chunk_ims,
                    chunk_height,
                    metrics,
                ),
            )

            if metrics.should_run_validation_partial_epoch():
                validation_partial_epoch(model, val_dataloader, metrics, "val_partial")
            if metrics.should_plot_losses():
                metrics.plot_metrics()

            print(f"Step time: {time.monotonic() - st:.2f}s")


@typechecked
def one_supervised_step(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    images: Float[Tensor, "slices channels height width"],
    gt_boundaries: Float[Tensor, "slices layers width"],
    metrics: Metrics,
) -> dict[str, float]:
    """Single training step on a volume chunk."""
    optimizer.zero_grad()

    T, C, H, W = images.shape

    # Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
    images_batch = images.unsqueeze(0)

    # Create ground truth masks for each slice
    gt_masks = torch.stack([
        heightmap_to_volume(gt_boundaries[t:t+1], (H, W))
        for t in range(T)
    ], dim=1).squeeze(2)  # (1, T, H, W)

    with torch.amp.autocast("cuda"):
        # Get predictions: (1, T, num_classes, H, W)
        logits = model(images_batch)

        # Reshape for cross entropy: (T, num_classes, H, W) and (T, H, W)
        logits_flat = logits.squeeze(0)  # (T, num_classes, H, W)
        gt_flat = gt_masks.squeeze(0)  # (T, H, W)

        loss = F.cross_entropy(logits_flat, gt_flat)

    if metrics.should_save_train_images():
        save_reconstruction_segmentation(
            images,
            gt_boundaries,
            logits_flat,
            f"selected_images/octcube_train_{metrics.current_iter}.png",
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    result = {"loss": loss.item()}
    result.update(compute_dice_scores(gt_boundaries, logits_flat, (H, W)))
    return result


def validation_epoch(model, dataloader, metrics: Metrics, split):
    """Full validation epoch."""
    from collections import defaultdict

    print(f"Running validation: {split}")
    model.eval()

    results = defaultdict(lambda: 0.0)
    num_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            ims = batch["frames"].permute(1, 0, 2, 3).cuda()
            height = batch["label"][0].permute(1, 0, 2).cuda()
            S, C, H, W = ims.shape

            for run in range(0, S, TrainConfig.step_size):
                end_idx = min(run + TrainConfig.step_size, S)
                chunk_size = end_idx - run

                # Pad if necessary
                if chunk_size % TrainConfig.t_patch_size != 0:
                    pad_size = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                    chunk_ims = ims[run:end_idx]
                    chunk_height = height[run:end_idx]
                    chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_size, -1, -1, -1)], dim=0)
                    chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_size, -1, -1)], dim=0)
                else:
                    chunk_ims = ims[run:end_idx]
                    chunk_height = height[run:end_idx]

                local_metrics = one_validation_step(
                    model,
                    chunk_ims,
                    chunk_height,
                    metrics,
                    num_steps == 0,
                    split,
                )
                for k, v in local_metrics.items():
                    results[k] += v
                num_steps += 1

    for k, v in results.items():
        results[k] = v / max(num_steps, 1)

    metrics.append(split, dict(results))
    metrics.print_latest(split)
    metrics.plot_metrics()


def validation_partial_epoch(model, dataloader, metrics: Metrics, split):
    """Partial validation on subset of data."""
    from collections import defaultdict

    print("Running Partial Validation")
    model.eval()

    results = defaultdict(lambda: 0.0)
    num_steps = 0
    max_vols = 10

    with torch.no_grad():
        for vol_idx, batch in enumerate(dataloader):
            if vol_idx >= max_vols:
                break

            ims = batch["frames"].permute(1, 0, 2, 3).cuda()
            height = batch["label"][0].permute(1, 0, 2).cuda()
            S, C, H, W = ims.shape

            # Just process first chunk for speed
            end_idx = min(TrainConfig.step_size, S)
            chunk_size = end_idx

            if chunk_size % TrainConfig.t_patch_size != 0:
                pad_size = TrainConfig.t_patch_size - (chunk_size % TrainConfig.t_patch_size)
                chunk_ims = ims[:end_idx]
                chunk_height = height[:end_idx]
                chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_size, -1, -1, -1)], dim=0)
                chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_size, -1, -1)], dim=0)
            else:
                chunk_ims = ims[:end_idx]
                chunk_height = height[:end_idx]

            local_metrics = one_validation_step(
                model,
                chunk_ims,
                chunk_height,
                metrics,
                vol_idx == 0,
                split,
            )
            for k, v in local_metrics.items():
                results[k] += v
            num_steps += 1

    for k, v in results.items():
        results[k] = v / max(num_steps, 1)

    metrics.append(split, dict(results))
    metrics.print_latest(split)
    metrics.plot_metrics()
    metrics.save("octcube_metrics.json")
    model.train()


def one_validation_step(
    model,
    images,
    gt_boundaries,
    metrics,
    save_images,
    split,
):
    """Single validation step on a volume chunk."""
    T, C, H, W = images.shape

    # Add batch dimension
    images_batch = images.unsqueeze(0)

    # Create ground truth masks
    gt_masks = torch.stack([
        heightmap_to_volume(gt_boundaries[t:t+1], (H, W))
        for t in range(T)
    ], dim=1).squeeze(2)

    with torch.no_grad():
        logits = model(images_batch)
        logits_flat = logits.squeeze(0)
        gt_flat = gt_masks.squeeze(0)
        loss = F.cross_entropy(logits_flat, gt_flat)

    if save_images:
        save_reconstruction_segmentation(
            images,
            gt_boundaries,
            logits_flat,
            f"selected_images/octcube_{split}_{metrics.current_iter}.png",
        )

    result = {"loss": loss.item()}
    result.update(compute_dice_scores(gt_boundaries, logits_flat, (H, W)))
    return result


def save_reconstruction_segmentation(images, gt_boundaries, logits, fname):
    """Save visualization of segmentation results."""
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    T, C, H, W = images.shape

    # Original (first slice)
    axes[0].imshow(images[0, 0].cpu().detach().numpy(), cmap="gray")
    axes[0].set_title(f"Original (slice 0/{T})")

    # GT segmentation
    segmentation_gt = heightmap_to_volume(gt_boundaries[:1], (H, W)).cpu().detach().numpy()
    axes[1].imshow(segmentation_gt[0], cmap="viridis")
    axes[1].set_title("Segmentation GT")

    # Predicted segmentation
    seg_pred = logits[0].argmax(0).cpu().detach().numpy().astype(int)
    axes[2].imshow(seg_pred, cmap="viridis")
    axes[2].set_title("Segmentation Pred")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def full_supervised_run():
    """Main training function."""
    os.makedirs("selected_images", exist_ok=True)
    os.makedirs(TrainConfig.save_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        SpectralisLoader(
            split_label="train",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=1,
        num_workers=5,
    )
    val_loader = torch.utils.data.DataLoader(
        SpectralisLoader(
            split_label="val",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=1,
        num_workers=5,
    )
    test_loader = torch.utils.data.DataLoader(
        SpectralisLoader(
            split_label="test",
            target_size=(TrainConfig.img_size, TrainConfig.img_size),
            normalize=True
        ),
        batch_size=1,
        num_workers=5,
    )
    print("Loaded data loaders")

    model = OCTCubeSegmenter(
        num_classes=TrainConfig.num_classes,
        img_size=TrainConfig.img_size,
        patch_size=TrainConfig.patch_size,
        num_frames=TrainConfig.num_frames,
        t_patch_size=TrainConfig.t_patch_size,
        size=TrainConfig.model_size,
        freeze_encoder=True,
        checkpoint_path=TrainConfig.checkpoint_path,
    ).cuda()
    print("Initialized Model")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scaler = GradScaler("cuda")
    print("Initialized Optimizer")

    metrics = Metrics()
    print("Initialized Metrics")

    # Track best validation performance
    best_val_dice = 0.0

    # Resume from checkpoint if specified
    start_epoch = 1
    if TrainConfig.resume_from is not None:
        if load_checkpoint(model, optimizer, scaler, metrics, TrainConfig.resume_from):
            start_epoch = metrics.current_epoch + 1
            # Load best dice from best checkpoint if it exists
            best_path = os.path.join(TrainConfig.save_dir, "best.pt")
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                best_val_dice = best_ckpt["metrics"]["data"]["val"]["metrics"].get("dice_mean", [0])[-1]
                print(f"Best validation dice so far: {best_val_dice:.4f}")

    latest_path = os.path.join(TrainConfig.save_dir, "latest.pt")
    best_path = os.path.join(TrainConfig.save_dir, "best.pt")

    for e in range(start_epoch, TrainConfig.epochs + 1):
        metrics.current_epoch = e
        print(f"Epoch: {e}")
        supervised_epoch(
            model,
            optimizer,
            scaler,
            train_loader,
            val_loader,
            metrics,
        )

        print("validation")
        validation_epoch(model, val_loader, metrics, "val")

        # Get current validation dice
        current_val_dice = metrics.get_latest_val_dice()

        # Save latest checkpoint
        save_checkpoint(model, optimizer, scaler, metrics, latest_path)

        # Save best checkpoint if this is the best so far
        if current_val_dice is not None and current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            save_checkpoint(model, optimizer, scaler, metrics, best_path, is_best=True)
            print(f"New best validation dice: {best_val_dice:.4f}")

    print("testing")
    validation_epoch(model, test_loader, metrics, "test")

    # Save final metrics
    metrics.save(os.path.join(TrainConfig.save_dir, "metrics_final.json"))


if __name__ == "__main__":
    full_supervised_run()
