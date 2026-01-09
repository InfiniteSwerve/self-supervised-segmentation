"""
Training script for OCTCube-based segmentation.

This script adapts the MIRAGE training pipeline to work with OCTCube,
which processes 3D OCT volumes instead of individual slices.

Two training modes are supported:
1. Volume mode: Process entire volumes at once (memory intensive)
2. Sliding window mode: Process windows of slices with temporal context
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import numpy as np
import einops
from spectralis_dataset import SpectralisLoader
import matplotlib.pyplot as plt
import time
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import os

from octcube import OCTCubeSegmenter, OCTCubeSliceSegmenter

typechecked = jaxtyped(typechecker=beartype)


@dataclass
class TrainConfig:
    # Training parameters
    step_size: int = 48  # Number of slices to process at once (should match num_frames)
    context_window: int = 5  # Context slices on each side for sliding window mode
    partial_val_interval: int = 500
    train_save_im: int = 300
    plot_losses: int = 300
    epochs: int = 20
    dice_calc_interval: int = 50

    # Model parameters
    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 48
    model_size: str = 'large'

    # Training mode: 'volume' or 'sliding_window'
    mode: str = 'sliding_window'

    # Paths
    checkpoint_path: Optional[str] = None
    save_dir: str = 'checkpoints_octcube'


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


@typechecked
def compute_dice_scores_volume(
    gt_boundaries: Float[Tensor, "batch slices layers width"],
    pred_vol: Float[Tensor, "batch slices num_classes height width"],
) -> dict[str, float]:
    """Compute dice scores for 3D volume predictions."""
    B, T, num_classes, H, W = pred_vol.shape

    # Flatten batch and time dimensions
    pred_flat = pred_vol.argmax(2).reshape(B * T, H, W)

    # Convert boundaries to masks for each slice
    gt_flat = []
    for b in range(B):
        for t in range(T):
            gt_mask = heightmap_to_volume(
                gt_boundaries[b, t:t+1], (H, W)
            )
            gt_flat.append(gt_mask)
    gt_flat = torch.cat(gt_flat, dim=0)

    # One-hot encode
    pred_oh = F.one_hot(pred_flat, num_classes).permute(0, 3, 1, 2).float()
    gt_oh = F.one_hot(gt_flat, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_oh * gt_oh).sum(dim=(0, 2, 3))
    union = pred_oh.sum(dim=(0, 2, 3)) + gt_oh.sum(dim=(0, 2, 3))

    dice = torch.where(union > 0, 2 * intersection / union, torch.ones_like(union))

    dice_scores = {f"dice_L{c}": dice[c].item() for c in range(num_classes)}
    dice_scores["dice_mean"] = dice.mean().item()
    return dice_scores


def resize_volume(images: Tensor, target_size: Tuple[int, int]) -> Tensor:
    """Resize a volume of images to target size."""
    B, T, C, H, W = images.shape
    if (H, W) == target_size:
        return images

    # Reshape for batch processing
    images = images.reshape(B * T, C, H, W)
    images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)
    images = images.reshape(B, T, C, *target_size)
    return images


def resize_boundaries(boundaries: Tensor, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tensor:
    """Resize boundary heightmaps to match target image size."""
    if original_size == target_size:
        return boundaries

    B, T, K, W = boundaries.shape
    scale_h = target_size[0] / original_size[0]
    scale_w = target_size[1] / original_size[1]

    # Scale boundary positions
    boundaries = boundaries * scale_h  # Scale heights

    # Resize width dimension
    boundaries = boundaries.reshape(B * T, K, W)
    boundaries = F.interpolate(
        boundaries.unsqueeze(1),
        size=(K, target_size[1]),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)
    boundaries = boundaries.reshape(B, T, K, target_size[1])

    return boundaries


########################################################################
# Volume-mode training (process entire volume at once)


def volume_supervised_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    train_dataloader,
    val_dataloader,
    metrics: Metrics,
    config: TrainConfig,
):
    """Training epoch for volume mode."""
    model.train()

    for batch in train_dataloader:
        st = time.monotonic()

        # Load volume: (C, S, H, W) -> (1, S, C, H, W)
        ims = batch["frames"].permute(0, 1, 2, 3).unsqueeze(0).cuda()  # (1, S, C, H, W)
        # Boundaries: (C, L, S, W) -> (1, S, L, W)
        height = batch["label"].permute(2, 0, 1, 3).unsqueeze(0).cuda()

        B, S, C, H, W = ims.shape

        # Resize if needed
        if H != config.img_size or W != config.img_size:
            ims = resize_volume(ims, (config.img_size, config.img_size))
            height = resize_boundaries(height, (H, W), (config.img_size, config.img_size))

        print(f"Data loading: {time.monotonic() - st:.2f}s")

        # Process in chunks of num_frames
        for start_idx in range(0, S, config.step_size):
            end_idx = min(start_idx + config.step_size, S)
            chunk_ims = ims[:, start_idx:end_idx]
            chunk_height = height[:, start_idx:end_idx]

            # Pad if necessary
            actual_frames = end_idx - start_idx
            if actual_frames < config.num_frames:
                pad_size = config.num_frames - actual_frames
                chunk_ims = F.pad(chunk_ims, (0, 0, 0, 0, 0, 0, 0, pad_size), mode='replicate')
                chunk_height = F.pad(chunk_height, (0, 0, 0, 0, 0, pad_size), mode='replicate')

            st = time.monotonic()
            metrics_dict = one_volume_step(
                model, optimizer, scaler,
                chunk_ims, chunk_height,
                actual_frames, metrics
            )
            metrics.append("train", metrics_dict)

            if metrics.should_run_validation_partial_epoch():
                validation_partial_epoch_volume(model, val_dataloader, metrics, config)
            if metrics.should_plot_losses():
                metrics.plot_metrics()

            print(f"Step time: {time.monotonic() - st:.2f}s")


def one_volume_step(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    images: Tensor,  # (B, T, C, H, W)
    gt_boundaries: Tensor,  # (B, T, L, W)
    valid_frames: int,
    metrics: Metrics,
) -> dict[str, float]:
    optimizer.zero_grad()

    B, T, C, H, W = images.shape

    # Create ground truth masks for each slice
    gt_masks = []
    for t in range(T):
        gt_mask = heightmap_to_volume(gt_boundaries[:, t], (H, W))
        gt_masks.append(gt_mask)
    gt_masks = torch.stack(gt_masks, dim=1)  # (B, T, H, W)

    with torch.amp.autocast("cuda"):
        logits = model(images)  # (B, T, num_classes, H, W)

        # Only compute loss on valid frames
        logits_valid = logits[:, :valid_frames]
        gt_masks_valid = gt_masks[:, :valid_frames]

        # Reshape for cross entropy
        logits_flat = logits_valid.reshape(-1, logits.shape[2], H, W)
        gt_flat = gt_masks_valid.reshape(-1, H, W)

        loss = F.cross_entropy(logits_flat, gt_flat)

    if metrics.should_save_train_images():
        save_volume_segmentation(
            images[:, :valid_frames],
            gt_boundaries[:, :valid_frames],
            logits[:, :valid_frames],
            f"selected_images/octcube_train_{metrics.current_iter}.png"
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    result = {"loss": loss.item()}
    result.update(compute_dice_scores_volume(
        gt_boundaries[:, :valid_frames],
        logits[:, :valid_frames]
    ))
    return result


########################################################################
# Sliding window mode training


def sliding_window_supervised_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    train_dataloader,
    val_dataloader,
    metrics: Metrics,
    config: TrainConfig,
):
    """Training epoch for sliding window mode."""
    model.train()

    for batch in train_dataloader:
        st = time.monotonic()

        # Load data: (C, S, H, W) -> (S, C, H, W)
        ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
        height = einops.rearrange(batch["label"], "c l s w -> s l w").cuda()

        S, C, H, W = ims.shape

        # Resize if needed
        if H != config.img_size or W != config.img_size:
            ims = F.interpolate(ims, size=(config.img_size, config.img_size), mode='bilinear', align_corners=False)
            scale = config.img_size / H
            height = height * scale
            # Resize width
            height = F.interpolate(
                height.unsqueeze(1),
                size=(height.shape[1], config.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        print(f"Data loading: {time.monotonic() - st:.2f}s")

        # Process in batches with context
        context = config.context_window
        step_size = config.step_size

        for start_idx in range(0, S, step_size):
            end_idx = min(start_idx + step_size, S)

            # Get context slices
            ctx_start = max(0, start_idx - context)
            ctx_end = min(S, end_idx + context)

            # Current slices to segment
            current_ims = ims[start_idx:end_idx]  # (T, C, H, W)
            current_height = height[start_idx:end_idx]  # (T, L, W)

            # Context before
            if start_idx > 0:
                context_before = ims[ctx_start:start_idx]
            else:
                context_before = ims[0:1].expand(context, -1, -1, -1)

            # Context after
            if end_idx < S:
                context_after = ims[end_idx:ctx_end]
            else:
                context_after = ims[-1:].expand(context, -1, -1, -1)

            # Pad context to exact size
            if context_before.shape[0] < context:
                pad_size = context - context_before.shape[0]
                context_before = F.pad(
                    context_before.unsqueeze(0),
                    (0, 0, 0, 0, 0, 0, pad_size, 0),
                    mode='replicate'
                ).squeeze(0)
            if context_after.shape[0] < context:
                pad_size = context - context_after.shape[0]
                context_after = F.pad(
                    context_after.unsqueeze(0),
                    (0, 0, 0, 0, 0, 0, 0, pad_size),
                    mode='replicate'
                ).squeeze(0)

            st = time.monotonic()
            metrics_dict = one_sliding_window_step(
                model, optimizer, scaler,
                current_ims, current_height,
                context_before, context_after,
                metrics
            )
            metrics.append("train", metrics_dict)

            if metrics.should_run_validation_partial_epoch():
                validation_partial_epoch_sliding(model, val_dataloader, metrics, config)
            if metrics.should_plot_losses():
                metrics.plot_metrics()

            print(f"Step time: {time.monotonic() - st:.2f}s")


def one_sliding_window_step(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    current_ims: Tensor,  # (T, C, H, W)
    gt_boundaries: Tensor,  # (T, L, W)
    context_before: Tensor,  # (ctx, C, H, W)
    context_after: Tensor,  # (ctx, C, H, W)
    metrics: Metrics,
) -> dict[str, float]:
    optimizer.zero_grad()

    T, C, H, W = current_ims.shape

    # Build volume: (context_before, current, context_after)
    volume = torch.cat([context_before, current_ims, context_after], dim=0)  # (T+2*ctx, C, H, W)
    volume = volume.unsqueeze(0)  # (1, T+2*ctx, C, H, W)

    # Ground truth masks
    gt_masks = []
    for t in range(T):
        gt_mask = heightmap_to_volume(gt_boundaries[t:t+1], (H, W))
        gt_masks.append(gt_mask)
    gt_masks = torch.cat(gt_masks, dim=0)  # (T, H, W)

    ctx = context_before.shape[0]

    with torch.amp.autocast("cuda"):
        # Get full volume prediction
        logits = model(volume)  # (1, T+2*ctx, num_classes, H, W)

        # Extract only the current slice predictions (center portion)
        logits_center = logits[0, ctx:ctx+T]  # (T, num_classes, H, W)

        loss = F.cross_entropy(logits_center, gt_masks)

    if metrics.should_save_train_images():
        save_slice_segmentation(
            current_ims,
            gt_boundaries,
            logits_center,
            f"selected_images/octcube_train_{metrics.current_iter}.png"
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    result = {"loss": loss.item()}
    result.update(compute_dice_scores(
        gt_boundaries, logits_center, (H, W)
    ))
    return result


########################################################################
# Validation


def validation_partial_epoch_volume(model, dataloader, metrics, config):
    """Partial validation for volume mode."""
    from collections import defaultdict

    print("Running Partial Validation (Volume Mode)")
    model.eval()

    results = defaultdict(float)
    num_batches = 0
    max_batches = 10

    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break

            ims = batch["frames"].permute(0, 1, 2, 3).unsqueeze(0).cuda()
            height = batch["label"].permute(2, 0, 1, 3).unsqueeze(0).cuda()

            B, S, C, H, W = ims.shape

            if H != config.img_size or W != config.img_size:
                ims = resize_volume(ims, (config.img_size, config.img_size))
                height = resize_boundaries(height, (H, W), (config.img_size, config.img_size))

            # Process first chunk only for speed
            chunk_ims = ims[:, :config.num_frames]
            chunk_height = height[:, :config.num_frames]

            logits = model(chunk_ims)

            # Compute metrics
            local_metrics = compute_dice_scores_volume(chunk_height, logits)
            for k, v in local_metrics.items():
                results[k] += v

            num_batches += 1

    for k in results:
        results[k] /= num_batches

    metrics.append("val_partial", dict(results))
    metrics.plot_metrics()
    metrics.save("octcube_metrics.json")
    model.train()


def validation_partial_epoch_sliding(model, dataloader, metrics, config):
    """Partial validation for sliding window mode."""
    from collections import defaultdict

    print("Running Partial Validation (Sliding Window Mode)")
    model.eval()

    results = defaultdict(float)
    num_steps = 0
    max_volumes = 10

    context = config.context_window

    with torch.no_grad():
        for vol_idx, batch in enumerate(dataloader):
            if vol_idx >= max_volumes:
                break

            ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
            height = einops.rearrange(batch["label"], "c l s w -> s l w").cuda()

            S, C, H, W = ims.shape

            if H != config.img_size or W != config.img_size:
                ims = F.interpolate(ims, size=(config.img_size, config.img_size), mode='bilinear', align_corners=False)
                scale = config.img_size / H
                height = height * scale
                height = F.interpolate(
                    height.unsqueeze(1),
                    size=(height.shape[1], config.img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                H, W = config.img_size, config.img_size

            # Process a few chunks
            for start_idx in range(0, min(S, config.step_size * 3), config.step_size):
                end_idx = min(start_idx + config.step_size, S)

                ctx_start = max(0, start_idx - context)
                ctx_end = min(S, end_idx + context)

                current_ims = ims[start_idx:end_idx]
                current_height = height[start_idx:end_idx]

                if start_idx > 0:
                    context_before = ims[ctx_start:start_idx]
                else:
                    context_before = ims[0:1].expand(context, -1, -1, -1)

                if end_idx < S:
                    context_after = ims[end_idx:ctx_end]
                else:
                    context_after = ims[-1:].expand(context, -1, -1, -1)

                if context_before.shape[0] < context:
                    pad_size = context - context_before.shape[0]
                    context_before = F.pad(
                        context_before.unsqueeze(0),
                        (0, 0, 0, 0, 0, 0, pad_size, 0),
                        mode='replicate'
                    ).squeeze(0)
                if context_after.shape[0] < context:
                    pad_size = context - context_after.shape[0]
                    context_after = F.pad(
                        context_after.unsqueeze(0),
                        (0, 0, 0, 0, 0, 0, 0, pad_size),
                        mode='replicate'
                    ).squeeze(0)

                volume = torch.cat([context_before, current_ims, context_after], dim=0)
                volume = volume.unsqueeze(0)

                logits = model(volume)
                T = current_ims.shape[0]
                logits_center = logits[0, context:context+T]

                local_metrics = compute_dice_scores(current_height, logits_center, (H, W))
                for k, v in local_metrics.items():
                    results[k] += v
                num_steps += 1

    for k in results:
        results[k] /= max(num_steps, 1)

    metrics.append("val_partial", dict(results))
    metrics.plot_metrics()
    metrics.save("octcube_metrics.json")
    model.train()


def validation_epoch(model, dataloader, metrics, split, config):
    """Full validation epoch."""
    from collections import defaultdict

    print(f"Running Full Validation ({split})")
    model.eval()

    results = defaultdict(float)
    num_steps = 0
    context = config.context_window

    with torch.no_grad():
        for batch in dataloader:
            ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
            height = einops.rearrange(batch["label"], "c l s w -> s l w").cuda()

            S, C, H, W = ims.shape

            if H != config.img_size or W != config.img_size:
                ims = F.interpolate(ims, size=(config.img_size, config.img_size), mode='bilinear', align_corners=False)
                scale = config.img_size / H
                height = height * scale
                height = F.interpolate(
                    height.unsqueeze(1),
                    size=(height.shape[1], config.img_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                H, W = config.img_size, config.img_size

            for start_idx in range(0, S, config.step_size):
                end_idx = min(start_idx + config.step_size, S)

                ctx_start = max(0, start_idx - context)
                ctx_end = min(S, end_idx + context)

                current_ims = ims[start_idx:end_idx]
                current_height = height[start_idx:end_idx]

                if start_idx > 0:
                    context_before = ims[ctx_start:start_idx]
                else:
                    context_before = ims[0:1].expand(context, -1, -1, -1)

                if end_idx < S:
                    context_after = ims[end_idx:ctx_end]
                else:
                    context_after = ims[-1:].expand(context, -1, -1, -1)

                if context_before.shape[0] < context:
                    pad_size = context - context_before.shape[0]
                    context_before = F.pad(
                        context_before.unsqueeze(0),
                        (0, 0, 0, 0, 0, 0, pad_size, 0),
                        mode='replicate'
                    ).squeeze(0)
                if context_after.shape[0] < context:
                    pad_size = context - context_after.shape[0]
                    context_after = F.pad(
                        context_after.unsqueeze(0),
                        (0, 0, 0, 0, 0, 0, 0, pad_size),
                        mode='replicate'
                    ).squeeze(0)

                volume = torch.cat([context_before, current_ims, context_after], dim=0)
                volume = volume.unsqueeze(0)

                logits = model(volume)
                T = current_ims.shape[0]
                logits_center = logits[0, context:context+T]

                local_metrics = compute_dice_scores(current_height, logits_center, (H, W))
                for k, v in local_metrics.items():
                    results[k] += v
                num_steps += 1

    for k in results:
        results[k] /= max(num_steps, 1)

    metrics.append(split, dict(results))
    metrics.plot_metrics()


########################################################################
# Visualization


def save_volume_segmentation(images, gt_boundaries, logits, fname):
    """Save visualization for volume predictions."""
    # Show first slice
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    B, T, C, H, W = images.shape

    # Original
    axes[0].imshow(images[0, 0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title(f"Original (slice 0/{T})")

    # GT segmentation
    gt_mask = heightmap_to_volume(gt_boundaries[0, 0:1], (H, W)).cpu().numpy()
    axes[1].imshow(gt_mask[0], cmap='viridis')
    axes[1].set_title("GT Segmentation")

    # Predicted segmentation
    pred_mask = logits[0, 0].argmax(0).cpu().numpy()
    axes[2].imshow(pred_mask, cmap='viridis')
    axes[2].set_title("Predicted Segmentation")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname)
    plt.close()


def save_slice_segmentation(images, gt_boundaries, logits, fname):
    """Save visualization for slice predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T, C, H, W = images.shape

    # Original
    axes[0].imshow(images[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title(f"Original (slice 0/{T})")

    # GT segmentation
    gt_mask = heightmap_to_volume(gt_boundaries[0:1], (H, W)).cpu().numpy()
    axes[1].imshow(gt_mask[0], cmap='viridis')
    axes[1].set_title("GT Segmentation")

    # Predicted segmentation
    pred_mask = logits[0].argmax(0).cpu().numpy()
    axes[2].imshow(pred_mask, cmap='viridis')
    axes[2].set_title("Predicted Segmentation")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname)
    plt.close()


########################################################################
# Main training functions


def full_supervised_run(config: TrainConfig):
    """Main training function."""

    # Create output directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs("selected_images", exist_ok=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="train", target_size=(config.img_size, config.img_size), normalize=True),
        batch_size=1,
        num_workers=4,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="val", target_size=(config.img_size, config.img_size), normalize=True),
        batch_size=1,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="test", target_size=(config.img_size, config.img_size), normalize=True),
        batch_size=1,
        num_workers=4,
    )
    print("Loaded data loaders")

    # Create model
    num_classes = 12  # 11 boundaries + 1 background

    if config.mode == 'volume':
        model = OCTCubeSegmenter(
            num_classes=num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            size=config.model_size,
            freeze_encoder=True,
            checkpoint_path=config.checkpoint_path,
        ).cuda()
    else:  # sliding_window
        total_frames = 2 * config.context_window + config.step_size
        model = OCTCubeSegmenter(
            num_classes=num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            num_frames=total_frames,
            size=config.model_size,
            freeze_encoder=True,
            checkpoint_path=config.checkpoint_path,
        ).cuda()

    print(f"Initialized OCTCube model ({config.mode} mode)")

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    print("Initialized optimizer")

    # Metrics
    metrics = Metrics()
    print("Initialized metrics")

    # Training loop
    for epoch in range(1, config.epochs + 1):
        metrics.current_epoch = epoch
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*50}")

        if config.mode == 'volume':
            volume_supervised_epoch(
                model, optimizer, scaler,
                train_loader, val_loader, metrics, config
            )
        else:
            sliding_window_supervised_epoch(
                model, optimizer, scaler,
                train_loader, val_loader, metrics, config
            )

        # Full validation
        print("\nRunning full validation...")
        validation_epoch(model, val_loader, metrics, "val", config)

        # Save checkpoint
        checkpoint_path = os.path.join(config.save_dir, f"octcube_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.data,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Test
    print("\n" + "="*50)
    print("Running test evaluation...")
    print("="*50)
    validation_epoch(model, test_loader, metrics, "test", config)

    # Save final metrics
    metrics.save("octcube_metrics_final.json")
    metrics.plot_metrics()

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train OCTCube segmentation model")
    parser.add_argument('--mode', type=str, default='sliding_window',
                        choices=['volume', 'sliding_window'],
                        help='Training mode: volume (full volume) or sliding_window (with context)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--num_frames', type=int, default=48, help='Number of frames for volume mode')
    parser.add_argument('--step_size', type=int, default=48, help='Step size for processing')
    parser.add_argument('--context_window', type=int, default=5, help='Context window for sliding mode')
    parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large'])
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints_octcube', help='Save directory')

    args = parser.parse_args()

    config = TrainConfig(
        mode=args.mode,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        step_size=args.step_size,
        context_window=args.context_window,
        model_size=args.model_size,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
    )

    print("Training Configuration:")
    print(f"  Mode: {config.mode}")
    print(f"  Image size: {config.img_size}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Model size: {config.model_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print()

    full_supervised_run(config)


if __name__ == "__main__":
    main()
