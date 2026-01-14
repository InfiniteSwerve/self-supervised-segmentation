"""
MAE-based anomaly detection for OCT volumes.

Simple approach: compute per-patch reconstruction loss and visualize
the patches with highest loss (most anomalous).

Usage:
    python analyze_mae.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json

from spectralis_dataset import SpectralisLoader
from octcube import OCTCubeMAE


@dataclass
class AnalysisConfig:
    # Model parameters
    img_size: int = 512
    patch_size: int = 16
    num_frames: int = 48
    t_patch_size: int = 3
    model_size: str = 'large'

    # Paths
    mae_checkpoint: str = "path/to/octcube_checkpoint.pth"
    output_dir: str = "mae_analysis"

    # Analysis settings
    split: str = "val"
    max_volumes: Optional[int] = 10
    top_k_per_volume: int = 25  # Top patches to show per volume


@torch.no_grad()
def analyze_volume(
    model: OCTCubeMAE,
    images: torch.Tensor,  # (S, C, H, W) on GPU
    config: AnalysisConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
    """
    Compute per-patch reconstruction loss for a volume.

    Returns:
        patch_losses: (T_patches, H_patches, W_patches)
        reconstructed: (S, C, H, W)
        top_patches: list of (loss, t_idx, h_idx, w_idx)
    """
    S, C, H, W = images.shape
    p = config.patch_size
    t = config.t_patch_size

    all_patch_losses = []
    all_reconstructed = []

    for start in range(0, S, config.num_frames):
        end = min(start + config.num_frames, S)
        chunk_size = end - start

        # Pad if needed for t_patch_size divisibility
        if chunk_size % t != 0:
            pad_size = t - (chunk_size % t)
            chunk = images[start:end]
            chunk = torch.cat([chunk, chunk[-1:].expand(pad_size, -1, -1, -1)], dim=0)
            actual_size = chunk_size
        else:
            chunk = images[start:end]
            actual_size = chunk_size

        # Run MAE
        chunk_batch = chunk.unsqueeze(0)  # (1, T, C, H, W)
        reconstructed_patches, _ = model(chunk_batch)
        target_patches = model.patchify(chunk_batch)

        # Per-patch MSE
        patch_losses = ((reconstructed_patches - target_patches) ** 2).mean(dim=-1)
        patch_losses = patch_losses.squeeze(0)  # (T_patches, L_patches)

        # Reshape to spatial grid
        T_patches = chunk.shape[0] // t
        H_patches = H // p
        W_patches = W // p
        patch_losses = patch_losses.reshape(T_patches, H_patches, W_patches)

        # Keep only non-padded
        actual_T_patches = actual_size // t
        all_patch_losses.append(patch_losses[:actual_T_patches].cpu())

        # Reconstruct
        reconstructed = model.unpatchify(reconstructed_patches, chunk.shape[0], H, W)
        reconstructed = reconstructed.squeeze(0)[:actual_size]
        all_reconstructed.append(reconstructed.cpu())

    patch_losses = torch.cat(all_patch_losses, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)

    # Find top-k anomalous patches
    flat_losses = patch_losses.flatten()
    top_k = min(config.top_k_per_volume, len(flat_losses))
    top_values, top_indices = torch.topk(flat_losses, top_k)

    top_patches = []
    T_p, H_p, W_p = patch_losses.shape
    for val, idx in zip(top_values.tolist(), top_indices.tolist()):
        t_idx = idx // (H_p * W_p)
        remainder = idx % (H_p * W_p)
        h_idx = remainder // W_p
        w_idx = remainder % W_p
        top_patches.append((val, t_idx, h_idx, w_idx))

    return patch_losses, reconstructed, top_patches


def visualize_top_patches(
    images: torch.Tensor,  # (S, C, H, W)
    reconstructed: torch.Tensor,  # (S, C, H, W)
    top_patches: List[Tuple],
    volume_id: str,
    output_path: str,
    config: AnalysisConfig,
):
    """Visualize top anomalous patches: original, reconstruction, difference."""
    p = config.patch_size
    t = config.t_patch_size
    n = len(top_patches)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (loss, t_idx, h_idx, w_idx) in enumerate(top_patches):
        # Get slice (middle of temporal patch)
        slice_idx = min(t_idx * t + t // 2, images.shape[0] - 1)

        # Patch bounds
        y0, y1 = h_idx * p, (h_idx + 1) * p
        x0, x1 = w_idx * p, (w_idx + 1) * p

        orig_slice = images[slice_idx, 0].numpy()
        recon_slice = reconstructed[slice_idx, 0].numpy()

        # Full slice with box
        axes[i, 0].imshow(orig_slice, cmap='gray')
        axes[i, 0].add_patch(plt.Rectangle((x0, y0), p, p, fill=False, edgecolor='red', linewidth=2))
        axes[i, 0].set_title(f"Slice {slice_idx}" if i == 0 else "")
        axes[i, 0].axis('off')

        # Original patch
        axes[i, 1].imshow(orig_slice[y0:y1, x0:x1], cmap='gray')
        axes[i, 1].set_title(f"Original" if i == 0 else "")
        axes[i, 1].axis('off')

        # Reconstructed patch
        axes[i, 2].imshow(recon_slice[y0:y1, x0:x1], cmap='gray')
        axes[i, 2].set_title(f"Reconstructed" if i == 0 else "")
        axes[i, 2].axis('off')

        # Difference
        diff = np.abs(orig_slice[y0:y1, x0:x1] - recon_slice[y0:y1, x0:x1])
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f"Diff (loss={loss:.4f})" if i == 0 else f"loss={loss:.4f}")
        axes[i, 3].axis('off')

        # Row label
        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=10)

    plt.suptitle(f"Top {n} Anomalous Patches - {volume_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_analysis(config: AnalysisConfig):
    """Run anomaly analysis on dataset."""
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading MAE model...")
    model = OCTCubeMAE(
        img_size=config.img_size,
        patch_size=config.patch_size,
        num_frames=config.num_frames,
        t_patch_size=config.t_patch_size,
        size=config.model_size,
        checkpoint_path=config.mae_checkpoint,
    ).to(device)
    model.eval()

    # Load dataset
    print(f"Loading {config.split} dataset...")
    loader = torch.utils.data.DataLoader(
        SpectralisLoader(
            split_label=config.split,
            target_size=(config.img_size, config.img_size),
            normalize=True,
        ),
        batch_size=1,
        num_workers=4,
    )

    # Track global stats
    all_top_patches = []  # (loss, volume_id, t, h, w)
    volume_stats = []

    print(f"\nAnalyzing volumes...")
    for vol_idx, batch in enumerate(loader):
        if config.max_volumes and vol_idx >= config.max_volumes:
            break

        # Get volume ID
        volume_id = batch.get("path", [f"vol_{vol_idx}"])[0]
        if isinstance(volume_id, (list, tuple)):
            volume_id = volume_id[0]
        volume_id = Path(volume_id).stem

        print(f"[{vol_idx + 1}] {volume_id}...", end=" ")

        # Load images
        images = batch["frames"].permute(1, 0, 2, 3).to(device)
        if images.shape[1] != 1:
            images = images[:, :1]  # Keep only first channel

        # Analyze
        patch_losses, reconstructed, top_patches = analyze_volume(model, images, config)

        # Stats
        stats = {
            "volume_id": volume_id,
            "loss_mean": float(patch_losses.mean()),
            "loss_max": float(patch_losses.max()),
            "loss_p95": float(torch.quantile(patch_losses.flatten(), 0.95)),
        }
        volume_stats.append(stats)
        print(f"mean={stats['loss_mean']:.4f}, max={stats['loss_max']:.4f}, p95={stats['loss_p95']:.4f}")

        # Track top patches globally
        for loss, t, h, w in top_patches:
            all_top_patches.append((loss, volume_id, t, h, w))

        # Visualize
        output_path = os.path.join(config.output_dir, f"{volume_id}_anomalies.png")
        visualize_top_patches(
            images.cpu(), reconstructed, top_patches,
            volume_id, output_path, config
        )

    # Global summary
    all_top_patches.sort(key=lambda x: x[0], reverse=True)
    volume_stats.sort(key=lambda x: x["loss_p95"], reverse=True)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Most anomalous volumes (by p95):")
    for s in volume_stats[:5]:
        print(f"  {s['volume_id']}: p95={s['loss_p95']:.4f}")

    print(f"\nTop 10 anomalous patches:")
    for loss, vid, t, h, w in all_top_patches[:10]:
        print(f"  {vid} (t={t}, h={h}, w={w}): {loss:.4f}")

    # Save summary
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump({
            "volume_stats": volume_stats,
            "top_patches": [
                {"loss": l, "volume": v, "t": t, "h": h, "w": w}
                for l, v, t, h, w in all_top_patches[:100]
            ]
        }, f, indent=2)

    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    config = AnalysisConfig(
        mae_checkpoint="path/to/octcube_checkpoint.pth",  # UPDATE THIS
        output_dir="mae_analysis",
        split="val",
        max_volumes=10,
        top_k_per_volume=25,
    )
    run_analysis(config)
