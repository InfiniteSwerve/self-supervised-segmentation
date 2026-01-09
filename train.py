import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
import cv2
import numpy as np
import einops
from spectralis_dataset import SpectralisLoader
from models import UNetEncoder, OCTSegmenter, SPADEDecoder
from shape_conversion_utils import *
import matplotlib.pyplot as plt
import time
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin
from mirage_hf import MIRAGEWrapper, MIRAGESegmenter

typechecked = jaxtyped(typechecker=beartype)


@dataclass
class TrainConfig:
    step_size: int = 70
    partial_val_interval: int = 1000
    train_save_im: int = 300
    plot_losses: int = 300
    epochs: int = 20
    dice_calc_interval: int = 50


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
        # get all unique metric names
        all_metrics = set()
        for split in self.data.values():
            all_metrics.update(split["metrics"].keys())

        fig, axes = plt.subplots(
            len(all_metrics), 1, figsize=(10, 4 * len(all_metrics))
        )
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
        plt.savefig("total_metrics.png")
        plt.close()

    def save(self, path):
        import json

        # Convert defaultdicts to regular dicts for JSON serialization
        save_data = {
            "current_iter": self.current_iter,
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
        for split in save_data["data"]:
            self.data[split]["iterations"] = save_data["data"][split]["iterations"]
            self.data[split]["metrics"] = defaultdict(
                list, save_data["data"][split]["metrics"]
            )

    def should_save_train_images(self):
        return self.current_iter % TrainConfig.train_save_im == 0

    def should_run_validation_partial_epoch(self):
        return self.current_iter % TrainConfig.partial_val_interval == 0

    def should_plot_losses(self):
        return self.current_iter % TrainConfig.plot_losses == 0

    def should_calc_dice(self):
        return self.current_iter % TrainConfig.dice_calc_interval == 0


def smoothness_loss(boundaries):
    # mean squared l->r adjacency height difference penalty
    return (boundaries[:, :, 1:] - boundaries[:, :, :-1]).pow(2).mean()


@typechecked
def heightmap_to_volume(
    heightmap: Float[Tensor, "batch layers width"], imshape, one_hot=False
):
    """
    heightmap: (B, K, W) tensor on GPU, may contain negative values for invalid
    Returns: (B, H, W) class indices, or (B, K+1, H, W) one-hot if one_hot=True
    """
    B, K, W = heightmap.shape

    # Treat negative as infinity (never "above" this boundary)
    boundaries = heightmap.clone()
    boundaries[boundaries < 0] = float("inf")

    y_coords = torch.arange(
        imshape[1], device=heightmap.device, dtype=heightmap.dtype
    ).view(1, 1, imshape[1], 1)
    boundaries = boundaries.unsqueeze(2)  # (B, K, 1, W)

    above_boundary = (y_coords >= boundaries).float()  # (B, K, H, W)
    mask_volume = above_boundary.sum(dim=1).long()  # (B, H, W)

    if one_hot:
        one_hot_volume = F.one_hot(mask_volume, num_classes=K + 1)  # (B, H, W, K+1)
        return one_hot_volume.permute(0, 3, 1, 2).float()  # (B, K+1, H, W)
    return mask_volume


@typechecked
def boundary_loss(
    pred: Float[Tensor, "batch layers width"],
    gt: Float[Tensor, "batch layers width"],
) -> Float[Tensor, ""]:  # scalar
    gt_clean = torch.where(torch.isnan(gt), pred.detach(), gt)
    return F.mse_loss(pred, gt_clean)

@typechecked
def compute_dice_scores(
    gt_boundaries: Float[Tensor, "batch layers width"],
    pred_vol: Float[Tensor, "batch num_classes height width"],
    imshape: tuple,
) -> dict[str, float]:

    num_classes = pred_vol.shape[1]
    pred_vol = pred_vol.argmax(1)  # (B, H, W)
    gt_vol = heightmap_to_volume(gt_boundaries, imshape)  # (B, H, W)

    # One-hot encode both: (B, H, W) -> (B, C, H, W)
    pred_oh = F.one_hot(pred_vol, num_classes).permute(0, 3, 1, 2).float()
    gt_oh = F.one_hot(gt_vol, num_classes).permute(0, 3, 1, 2).float()

    # Compute intersection and union per class: sum over B, H, W
    intersection = (pred_oh * gt_oh).sum(dim=(0, 2, 3))  # (C,)
    union = pred_oh.sum(dim=(0, 2, 3)) + gt_oh.sum(dim=(0, 2, 3))  # (C,)

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
    for batch in train_dataloader:
        st = time.monotonic()
        ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
        height = einops.rearrange(batch["label"], "c l s w -> c s l w").cuda()
        frame_size = batch["frames"][0].shape[0]
        print(f"Ims + height cuda: {time.monotonic() - st}")

        for run in range(0, frame_size, TrainConfig.step_size):
            st = time.monotonic()
            metrics.append(
                "train",
                one_supervised_step(
                    model,
                    optimizer,
                    scaler,
                    ims[run : run + TrainConfig.step_size],
                    height[0, run : run + TrainConfig.step_size],
                    metrics,
                ),
            )
            if metrics.should_run_validation_partial_epoch():
                validation_partial_epoch(
                    model, val_dataloader, metrics, 70, "val_partial"
                )
            if metrics.should_plot_losses():
                metrics.plot_metrics()
            print(f"iter time: {time.monotonic() - st}")


@typechecked
def one_supervised_step(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    images: Float[Tensor, "batch channels height width"],
    gt_boundaries: Float[Tensor, "batch layers width"],
    metrics: Metrics,
) -> dict[str, float]:
    optimizer.zero_grad()

    gt_mask = heightmap_to_volume(gt_boundaries, images[:, 0].shape)

    with torch.amp.autocast("cuda"):
        #boundaries, texture = model.encoder(images)
        #gt_reconstruction = model.decoder(gt_mask, texture)
        logits = model(images)
        #l2 = F.mse_loss(gt_reconstruction, images)
        #l3 = boundary_loss(boundaries, gt_boundaries)
        #loss = l2 + l3
        loss = F.cross_entropy(logits, gt_mask)



    if metrics.should_save_train_images():
        save_reconstruction_segmentation(
            images,
            gt_boundaries,
            logits,
            f"selected_images/train_{metrics.current_iter}.png",
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    result = {
        "loss": loss.item(),
    }
    #if metrics.should_calc_dice():
    result.update(compute_dice_scores(
        gt_boundaries, logits, images[:, 0].shape
    ))
    return result

    return {
        "loss": loss.item(),
        "super_decoder": l2.item(),
        "super_encoder": l3.item(),
    }


def validation_epoch(model, dataloader, metrics: Metrics, split):
    from collections import defaultdict

    results = defaultdict(lambda: 0.0)
    i = 0
    for batch in dataloader:
        ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
        height = einops.rearrange(batch["label"], "c l s w -> c s l w").cuda()
        frame_size = batch["frames"][0].shape[0]
        for run in range(0, frame_size, TrainConfig.step_size):
            local_metrics = one_validation_step(
                model,
                ims[run : run + TrainConfig.step_size],
                height[0, run : run + TrainConfig.step_size],
                metrics,
                i == 0,
                split,
                self_super_lambda=0.9,
            )
            for k, v in local_metrics.items():
                results[k] += v
            i += 1

    for k, v in results.items():
        results[k] = v / i
    metrics.append(split, results)
    metrics.print_latest(split)
    metrics.plot_metrics()


def validation_partial_epoch(model, dataloader, metrics: Metrics, step_size, split):
    from collections import defaultdict

    print("Running Partial Validation")

    results = defaultdict(lambda: 0.0)
    i = 0
    vols = 0
    for batch in dataloader:
        if vols == 20:
            break
        vols += 1

        ims = einops.rearrange(batch["frames"], "c s h w -> s c h w").cuda()
        height = einops.rearrange(batch["label"], "c l s w -> c s l w").cuda()
        frame_size = batch["frames"][0].shape[0]
        for run in range(0, frame_size, step_size):
            local_metrics = one_validation_step(
                model,
                ims[run : run + step_size],
                height[0, run : run + step_size],
                metrics,
                vols == 1,
                split,
                self_super_lambda=0.9,
            )
            for k, v in local_metrics.items():
                results[k] += v
            i += 1

    for k, v in results.items():
        results[k] = v / i
    metrics.append(split, results)
    metrics.print_latest(split)
    metrics.plot_metrics()
    metrics.save("metrics.json")


def one_validation_step(
    model,
    images,
    gt_boundaries,
    metrics,
    save_images,
    split,
    self_super_lambda,
    smooth_weight=0.0,
    save_seg=False,
):

    gt_mask = heightmap_to_volume(gt_boundaries, images[:, 0].shape, one_hot=True)

    with torch.no_grad():

        logits = model(images)
        loss = F.cross_entropy(logits, gt_mask)

    if save_images:
        save_reconstruction_segmentation(
            images,
            gt_boundaries,
            logits,
            f"selected_images/{split}_{metrics.current_iter}.png",
        )

    result = {
        "loss": loss.item(),
    }
    result.update(compute_dice_scores(
        gt_boundaries, logits, images[:, 0].shape
    ))
    return result

    return {
        "loss": loss.item(),
        "super_decoder": l2.item(),
        "super_encoder": l3.item(),
    }


def save_reconstruction_segmentation(
    images,
    gt_boundaries,
    logits,
    fname,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    # Original
    axes[0].imshow(images[0, 0].cpu().detach().numpy(), cmap="gray")
    axes[0].set_title("Original")

    # GT segmentation
    segmentation_gt = (
        heightmap_to_volume(gt_boundaries, images[:, 0].shape).cpu().detach().numpy()
    )
    axes[1].imshow(segmentation_gt[0], cmap="viridis")
    axes[1].set_title("Segmentation GT")

    # Predicted segmentation
    seg_pred = (
        logits.argmax(1)
        .cpu()
        .detach()
        .numpy()
        .astype(int)
    )
    axes[2].imshow(seg_pred[0], cmap="viridis")
    axes[2].set_title("Segmentation Pred")


    # Could put difference or argmax segmentation in the 6th slot
    # seg_argmax = (torch.argmax(segmentation, dim=1)[0]).cpu().detach().numpy()
    # axes[1, 2].imshow(seg_argmax, cmap="viridis")
    # axes[1, 2].set_title("Segmentation (argmax)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        modalities='bscan',
        size='base',
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            size=size,
        )


def full_supervised_run():
    train_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="train", target_size=(512,512), normalize=True), batch_size=1, num_workers=5
    )
    val_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="val"), batch_size=1, num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        SpectralisLoader(split_label="test"), batch_size=1, num_workers=5
    )
    print("loaded loaders")
    num_layers = 11
    #model = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Base")
    model = MIRAGESegmenter(num_classes=12).cuda()    # model = model.cuda()
    #model = torch.compile(model).cuda()
    print("Initialized Model")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda")
    print("Initialized Optimizer")
    metrics = Metrics()
    print("Initialized Metrics")

    for e in range(1, TrainConfig.epochs + 1):
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

    print("testing")
    validation_epoch(model, test_loader, metrics, "test")


if __name__ == "__main__":
    full_supervised_run()
