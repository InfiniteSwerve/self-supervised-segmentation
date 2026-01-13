"""
Inference script for OCTCube segmentation.

Loads a trained checkpoint and saves sample segmentation visualizations
from each volume in the dataset.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from spectralis_dataset import SpectralisLoader
from octcube import OCTCubeSegmenter


@dataclass
class InferenceConfig:
    # Model parameters (must match training)
    img_size: int = 512
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'
    num_classes: int = 12

    # Paths
    checkpoint_path: str = "checkpoints_octcube/best.pt"  # Trained segmentation head
    encoder_checkpoint: Optional[str] = None  # OCTCube pretrained encoder (if needed)
    output_dir: str = "inference_output"

    # Inference settings
    split: str = "test"  # Which split to run on: train, val, test
    samples_per_volume: int = 5  # Number of slices to save per volume
    max_volumes: Optional[int] = None  # Limit number of volumes (None = all)
    save_numpy: bool = False  # Also save raw predictions as .npy


def load_model(config: InferenceConfig) -> OCTCubeSegmenter:
    """Load model with trained weights."""
    print(f"Creating model...")
    model = OCTCubeSegmenter(
        num_classes=config.num_classes,
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_frames=config.num_frames,
        t_patch_size=config.t_patch_size,
        size=config.model_size,
        freeze_encoder=True,
        checkpoint_path=config.encoder_checkpoint,
    ).cuda()

    print(f"Loading checkpoint from {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location="cuda", weights_only=False)

    # Load segmentation head weights
    model.head.load_state_dict(checkpoint["head_state_dict"])
    print(f"  Loaded head weights")

    # Print checkpoint info
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"  Checkpoint from epoch {metrics.get('current_epoch', '?')}, iter {metrics.get('current_iter', '?')}")
        if "data" in metrics and "val" in metrics["data"]:
            val_dice = metrics["data"]["val"]["metrics"].get("dice_mean", [])
            if val_dice:
                print(f"  Validation dice: {val_dice[-1]:.4f}")

    model.eval()
    return model


def heightmap_to_volume(heightmap, imshape):
    """Convert boundary heightmap to volume mask."""
    B, K, W = heightmap.shape
    boundaries = heightmap.clone()
    boundaries[boundaries < 0] = float("inf")

    y_coords = torch.arange(
        imshape[1], device=heightmap.device, dtype=heightmap.dtype
    ).view(1, 1, imshape[1], 1)
    boundaries = boundaries.unsqueeze(2)

    above_boundary = (y_coords >= boundaries).float()
    mask_volume = above_boundary.sum(dim=1).long()
    return mask_volume


def compute_dice(pred, gt, num_classes):
    """Compute per-class dice scores."""
    pred_oh = F.one_hot(pred, num_classes).float()
    gt_oh = F.one_hot(gt, num_classes).float()

    intersection = (pred_oh * gt_oh).sum(dim=(0, 1))
    union = pred_oh.sum(dim=(0, 1)) + gt_oh.sum(dim=(0, 1))

    dice = torch.where(union > 0, 2 * intersection / union, torch.ones_like(union))
    return dice


def save_slice_visualization(
    image: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    slice_idx: int,
    volume_id: str,
    output_path: str,
    dice_scores: Optional[np.ndarray] = None,
):
    """Save visualization of a single slice."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original (slice {slice_idx})")
    axes[0].axis("off")

    # Ground truth segmentation
    axes[1].imshow(gt_seg, cmap="viridis", vmin=0, vmax=11)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted segmentation
    axes[2].imshow(pred_seg, cmap="viridis", vmin=0, vmax=11)
    title = "Prediction"
    if dice_scores is not None:
        title += f" (dice={dice_scores.mean():.3f})"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.suptitle(f"Volume: {volume_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_volume_montage(
    images: np.ndarray,
    gt_segs: np.ndarray,
    pred_segs: np.ndarray,
    slice_indices: list,
    volume_id: str,
    output_path: str,
    dice_mean: float,
):
    """Save montage of multiple slices from a volume."""
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))

    if n_slices == 1:
        axes = axes.reshape(3, 1)

    for i, idx in enumerate(slice_indices):
        # Original
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title(f"Slice {idx}")
        axes[0, i].axis("off")

        # GT
        axes[1, i].imshow(gt_segs[i], cmap="viridis", vmin=0, vmax=11)
        if i == 0:
            axes[1, i].set_ylabel("Ground Truth", fontsize=12)
        axes[1, i].axis("off")

        # Pred
        axes[2, i].imshow(pred_segs[i], cmap="viridis", vmin=0, vmax=11)
        if i == 0:
            axes[2, i].set_ylabel("Prediction", fontsize=12)
        axes[2, i].axis("off")

    plt.suptitle(f"{volume_id} | Mean Dice: {dice_mean:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_inference(config: InferenceConfig):
    """Run inference on dataset and save visualizations."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Load model
    model = load_model(config)

    # Load dataset
    print(f"\nLoading {config.split} dataset...")
    loader = torch.utils.data.DataLoader(
        SpectralisLoader(
            split_label=config.split,
            target_size=(config.img_size, config.img_size),
            normalize=True,
        ),
        batch_size=1,
        num_workers=4,
    )

    # Track overall metrics
    all_dice_scores = []
    volume_results = []

    print(f"\nRunning inference...")
    for vol_idx, batch in enumerate(loader):
        if config.max_volumes is not None and vol_idx >= config.max_volumes:
            break

        # Get volume ID
        volume_id = batch.get("path", [f"vol_{vol_idx}"])[0]
        if isinstance(volume_id, (list, tuple)):
            volume_id = volume_id[0]
        volume_id = os.path.basename(str(volume_id))
        volume_id_clean = Path(volume_id).stem  # Remove extension

        print(f"\n[{vol_idx + 1}] Processing {volume_id}...")

        # Load data
        ims = batch["frames"].permute(1, 0, 2, 3).cuda()  # (S, C, H, W)
        height = batch["label"][0].permute(1, 0, 2).cuda()  # (S, L, W)

        S, C, H, W = ims.shape
        print(f"  Volume shape: {S} slices, {H}x{W}")

        # Process in chunks
        all_preds = []
        all_gts = []

        for run in range(0, S, config.num_frames):
            end_idx = min(run + config.num_frames, S)
            chunk_size = end_idx - run

            # Pad if necessary
            if chunk_size % config.t_patch_size != 0:
                pad_size = config.t_patch_size - (chunk_size % config.t_patch_size)
                chunk_ims = ims[run:end_idx]
                chunk_height = height[run:end_idx]
                chunk_ims = torch.cat([chunk_ims, chunk_ims[-1:].expand(pad_size, -1, -1, -1)], dim=0)
                chunk_height = torch.cat([chunk_height, chunk_height[-1:].expand(pad_size, -1, -1)], dim=0)
                actual_size = chunk_size
            else:
                chunk_ims = ims[run:end_idx]
                chunk_height = height[run:end_idx]
                actual_size = chunk_size

            # Run model
            chunk_ims_batch = chunk_ims.unsqueeze(0)  # (1, T, C, H, W)
            logits = model(chunk_ims_batch)  # (1, T, num_classes, H, W)
            preds = logits.squeeze(0).argmax(dim=1)  # (T, H, W)

            # Get GT masks
            gt_masks = torch.stack([
                heightmap_to_volume(chunk_height[t:t+1], (H, W))
                for t in range(len(chunk_height))
            ], dim=0).squeeze(1)  # (T, H, W)

            # Only keep non-padded slices
            all_preds.append(preds[:actual_size])
            all_gts.append(gt_masks[:actual_size])

        # Concatenate all chunks
        all_preds = torch.cat(all_preds, dim=0)  # (S, H, W)
        all_gts = torch.cat(all_gts, dim=0)  # (S, H, W)

        # Compute dice scores
        dice = compute_dice(all_preds, all_gts, config.num_classes)
        dice_mean = dice.mean().item()
        all_dice_scores.append(dice_mean)

        print(f"  Dice: {dice_mean:.4f} (per-class: {dice.cpu().numpy().round(3)})")

        # Select slices to save (evenly spaced)
        if config.samples_per_volume >= S:
            slice_indices = list(range(S))
        else:
            slice_indices = np.linspace(0, S - 1, config.samples_per_volume, dtype=int).tolist()

        # Collect samples
        sample_images = []
        sample_gts = []
        sample_preds = []

        for idx in slice_indices:
            sample_images.append(ims[idx, 0].cpu().numpy())
            sample_gts.append(all_gts[idx].cpu().numpy())
            sample_preds.append(all_preds[idx].cpu().numpy())

        # Save montage
        montage_path = os.path.join(config.output_dir, f"{volume_id_clean}_montage.png")
        save_volume_montage(
            np.array(sample_images),
            np.array(sample_gts),
            np.array(sample_preds),
            slice_indices,
            volume_id_clean,
            montage_path,
            dice_mean,
        )
        print(f"  Saved montage: {montage_path}")

        # Optionally save individual slices
        vol_output_dir = os.path.join(config.output_dir, volume_id_clean)
        os.makedirs(vol_output_dir, exist_ok=True)

        for i, idx in enumerate(slice_indices):
            slice_path = os.path.join(vol_output_dir, f"slice_{idx:03d}.png")
            save_slice_visualization(
                sample_images[i],
                sample_gts[i],
                sample_preds[i],
                idx,
                volume_id_clean,
                slice_path,
                dice.cpu().numpy(),
            )

        # Optionally save numpy arrays
        if config.save_numpy:
            np.save(os.path.join(vol_output_dir, "predictions.npy"), all_preds.cpu().numpy())
            np.save(os.path.join(vol_output_dir, "ground_truth.npy"), all_gts.cpu().numpy())

        volume_results.append({
            "volume_id": volume_id_clean,
            "dice_mean": dice_mean,
            "dice_per_class": dice.cpu().numpy().tolist(),
            "num_slices": S,
        })

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Processed {len(volume_results)} volumes")
    print(f"Mean Dice: {np.mean(all_dice_scores):.4f} (+/- {np.std(all_dice_scores):.4f})")
    print(f"Min Dice: {np.min(all_dice_scores):.4f}")
    print(f"Max Dice: {np.max(all_dice_scores):.4f}")

    # Sort by dice to show hardest/easiest
    sorted_results = sorted(volume_results, key=lambda x: x["dice_mean"])

    print(f"\nHardest volumes:")
    for r in sorted_results[:5]:
        print(f"  {r['volume_id']}: {r['dice_mean']:.4f}")

    print(f"\nEasiest volumes:")
    for r in sorted_results[-5:]:
        print(f"  {r['volume_id']}: {r['dice_mean']:.4f}")

    print("=" * 60)

    # Save results summary
    import json
    summary_path = os.path.join(config.output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "split": config.split,
                "checkpoint_path": config.checkpoint_path,
                "num_volumes": len(volume_results),
            },
            "overall": {
                "dice_mean": float(np.mean(all_dice_scores)),
                "dice_std": float(np.std(all_dice_scores)),
                "dice_min": float(np.min(all_dice_scores)),
                "dice_max": float(np.max(all_dice_scores)),
            },
            "volumes": volume_results,
        }, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    config = InferenceConfig(
        checkpoint_path="checkpoints_octcube/best.pt",
        output_dir="inference_output",
        split="test",
        samples_per_volume=5,
        max_volumes=None,
        save_numpy=False,
    )
    run_inference(config)
