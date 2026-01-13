"""
OCTCube: 3D Vision Transformer for OCT Volume Processing

This module implements a wrapper around OCTCube, a 3D foundation model for
optical coherence tomography that processes entire volumes rather than
individual slices (like MIRAGE).

Reference: https://github.com/ZucksLiu/OCTCubeM
Paper: "OCTCube: a 3D foundation model for optical coherence tomography" (arXiv:2408.11227)
"""

from functools import partial
import math
from typing import Optional, Tuple, Mapping, Any
import warnings

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


########################################################################
# Utility functions


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sine-cosine positional embeddings.

    Args:
        embed_dim: embedding dimension
        grid_size: int or tuple of (H, W)
        cls_token: whether to include a position for cls token

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, H, W)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1. / 10000 ** (omega / (embed_dim / 2))

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_pos_embed(embed_dim, length):
    """Generate 1D sine-cosine positional embeddings for temporal dimension."""
    pos = np.arange(length, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1. / 10000 ** (omega / (embed_dim / 2))

    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings for different input sizes."""
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)


def interpolate_temporal_pos_embed(model, checkpoint_model, new_num_frames, t_patch_size=1):
    """Interpolate temporal position embeddings for different number of frames."""
    if 'pos_embed_temporal' in checkpoint_model:
        pos_embed_temporal = checkpoint_model['pos_embed_temporal']
        old_num_temporal = pos_embed_temporal.shape[1]
        new_num_temporal = new_num_frames // t_patch_size

        if old_num_temporal != new_num_temporal:
            print(f"Temporal position interpolate from {old_num_temporal} to {new_num_temporal}")
            pos_embed_temporal = pos_embed_temporal.permute(0, 2, 1)  # (1, D, T)
            pos_embed_temporal = F.interpolate(pos_embed_temporal, size=new_num_temporal, mode='linear')
            checkpoint_model['pos_embed_temporal'] = pos_embed_temporal.permute(0, 2, 1)


########################################################################
# Core building blocks


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            dropout_p=self.attn_drop if self.training else 0.0
        ).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
        attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


########################################################################
# Patch Embedding for 3D volumes


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for OCT volumes.

    Handles input of shape (B, T, C, H, W) where T is the number of slices.
    Uses 3D convolution for joint spatial-temporal patching (matching OCTCube checkpoint).
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 1024,
        num_frames: int = 48,
        t_patch_size: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.grid_size = img_size // patch_size
        self.num_patches_spatial = self.grid_size * self.grid_size
        self.num_patches_temporal = num_frames // t_patch_size
        self.num_patches = self.num_patches_spatial * self.num_patches_temporal

        # 3D convolution for joint spatial-temporal patching (matches OCTCube checkpoint)
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(t_patch_size, patch_size, patch_size),
            stride=(t_patch_size, patch_size, patch_size)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) tensor
        Returns:
            (B, T', L, D) tensor where T' = T / t_patch_size, L = spatial patches
        """
        B, T, C, H, W = x.shape

        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # (B, D, T', H', W')

        # Get actual output dimensions
        _, D, T_out, H_out, W_out = x.shape

        # (B, D, T', H', W') -> (B, T', H'*W', D)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, T_out, H_out * W_out, D)

        return x


########################################################################
# OCTCube Vision Transformer


class OCTCubeViT(nn.Module):
    """
    Spatial-Temporal Vision Transformer for OCT volumes.

    Based on the OCTCube architecture that processes 3D volumes with
    separate positional embeddings for spatial and temporal dimensions.

    Default parameters match the OCTCube checkpoint:
    - img_size=512, patch_size=16 -> 32x32 = 1024 spatial patches
    - t_patch_size=3 for temporal patching
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        num_frames: int = 48,
        t_patch_size: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer=None,
        sep_pos_embed: bool = True,
        cls_embed: bool = True,
        global_pool: bool = True,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.global_pool = global_pool

        # Patch embedding (uses 3D conv matching checkpoint)
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
        )

        self.grid_size = img_size // patch_size
        self.num_patches_spatial = self.grid_size * self.grid_size  # 32*32 = 1024 for 512/16
        self.num_patches_temporal = num_frames // t_patch_size  # 48/3 = 16

        # Class token
        if cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings (separate for spatial and temporal)
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_patches_spatial, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_patches_temporal, embed_dim)
            )
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            total_patches = self.num_patches_spatial * self.num_patches_temporal
            if cls_embed:
                total_patches += 1
            self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # For global pooling
        if global_pool:
            self.fc_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'pos_embed_spatial', 'pos_embed_temporal', 'cls_token'}

    def forward_features(self, x, return_all_tokens=False):
        """
        Args:
            x: (B, T, C, H, W) tensor
            return_all_tokens: if True, return all tokens instead of pooled

        Returns:
            If return_all_tokens:
                (B, T, L, D) spatial tokens per slice
            Else:
                (B, D) pooled features
        """
        B, T, C, H, W = x.shape

        # Patch embedding: (B, T, C, H, W) -> (B, T, L, D)
        x = self.patch_embed(x)
        _, T_patches, L, D = x.shape

        # Add positional embeddings
        if self.sep_pos_embed:
            # Interpolate spatial pos embed if needed
            if L != self.num_patches_spatial:
                pos_embed_spatial = F.interpolate(
                    self.pos_embed_spatial.reshape(1, self.grid_size, self.grid_size, D).permute(0, 3, 1, 2),
                    size=(int(L**0.5), int(L**0.5)),
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1).reshape(1, L, D)
            else:
                pos_embed_spatial = self.pos_embed_spatial

            # Interpolate temporal pos embed if needed
            if T_patches != self.num_patches_temporal:
                pos_embed_temporal = F.interpolate(
                    self.pos_embed_temporal.permute(0, 2, 1),
                    size=T_patches,
                    mode='linear'
                ).permute(0, 2, 1)
            else:
                pos_embed_temporal = self.pos_embed_temporal

            # Add spatial pos embed to each slice
            x = x + pos_embed_spatial.unsqueeze(1)  # (B, T, L, D)
            # Add temporal pos embed
            x = x + pos_embed_temporal.unsqueeze(2)  # (B, T, L, D)

        # Flatten spatial-temporal: (B, T, L, D) -> (B, T*L, D)
        x = rearrange(x, 'b t l d -> b (t l) d')

        # Add class token
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        if not self.sep_pos_embed:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if return_all_tokens:
            # Remove cls token and reshape back to (B, T, L, D)
            if self.cls_embed:
                x = x[:, 1:]
            x = rearrange(x, 'b (t l) d -> b t l d', t=T_patches, l=L)
            return x
        else:
            if self.global_pool:
                # Global average pooling (excluding cls token)
                if self.cls_embed:
                    x = x[:, 1:].mean(dim=1)
                else:
                    x = x.mean(dim=1)
                x = self.fc_norm(x)
            else:
                # Use cls token
                x = x[:, 0]
            return x

    def forward(self, x, return_all_tokens=False):
        return self.forward_features(x, return_all_tokens=return_all_tokens)


########################################################################
# OCTCube Wrapper


class OCTCubeWrapper(nn.Module):
    """
    Wrapper for OCTCube model with easy configuration and weight loading.

    Default parameters match the OCTCube checkpoint:
    - img_size=512, patch_size=16 -> 32x32 = 1024 spatial patches
    - t_patch_size=3, num_frames=48 -> 16 temporal patches
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 1,
        num_frames: int = 48,
        t_patch_size: int = 3,
        size: str = 'large',  # 'base' or 'large'
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size
        self.size = size

        if size == 'large':
            self.model = OCTCubeViT(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_frames=num_frames,
                t_patch_size=t_patch_size,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_path_rate=drop_path_rate,
            )
            self.embed_dim = 1024
        elif size == 'base':
            self.model = OCTCubeViT(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_frames=num_frames,
                t_patch_size=t_patch_size,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_path_rate=drop_path_rate,
            )
            self.embed_dim = 768
        else:
            raise ValueError(f"Unknown model size: {size}")

        print(f"Created OCTCube-{size} model with img_size={img_size}, "
              f"patch_size={patch_size}, num_frames={num_frames}, t_patch_size={t_patch_size}")

    def forward(self, x, return_all_tokens=False):
        """
        Args:
            x: (B, T, C, H, W) tensor of OCT volume
            return_all_tokens: if True, return spatial tokens per slice

        Returns:
            If return_all_tokens:
                (B, T, L, D) tensor of spatial tokens per slice
            Else:
                (B, D) pooled features
        """
        return self.model(x, return_all_tokens=return_all_tokens)

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights with position embedding interpolation."""
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Interpolate position embeddings if needed
        interpolate_pos_embed(self.model, state_dict)
        interpolate_temporal_pos_embed(
            self.model, state_dict,
            self.num_frames, self.t_patch_size
        )

        msg = self.model.load_state_dict(state_dict, strict=strict)
        print(f"  # Missing keys: {len(msg.missing_keys)}")
        print(f"  # Unexpected keys: {len(msg.unexpected_keys)}")

        # Print some examples to help debug
        if msg.missing_keys:
            print(f"  Missing keys (first 10): {msg.missing_keys[:10]}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys (first 10): {msg.unexpected_keys[:10]}")
            # Check if decoder keys are present (MAE decoder)
            decoder_keys = [k for k in msg.unexpected_keys if 'decoder' in k.lower()]
            print(f"  Decoder keys in checkpoint: {len(decoder_keys)}")

        return msg

    @property
    def device(self):
        return next(self.parameters()).device


########################################################################
# ConvNeXt-based Segmentation Head


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for the segmentation head."""

    def __init__(self, dim=96, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x + residual


class OCTCubeSegHead(nn.Module):
    """
    Segmentation head for OCTCube that produces per-slice segmentations.

    Takes tokens from OCTCube and upsamples to produce full-resolution
    segmentation masks for each temporal patch group.

    Note: OCTCube uses t_patch_size=3, so T input slices become T/3 temporal tokens.
    The head produces T/3 segmentation maps which need to be upsampled temporally
    if per-slice output is needed.
    """

    def __init__(
        self,
        num_classes: int,
        in_dim: int = 1024,
        hidden_dim: int = 96,
        patch_grid: Tuple[int, int] = (32, 32),  # 512/16 = 32
        num_blocks: int = 4,
    ):
        super().__init__()

        self.patch_grid = patch_grid
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        h, w = patch_grid
        # Each token unfolds to 8x8 spatial region
        unfold_size = 8
        self.proj = nn.Linear(in_dim, hidden_dim * unfold_size * unfold_size)

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(hidden_dim, expansion=4) for _ in range(num_blocks)
        ])

        # Upsample from (h*8, w*8) = (256, 256) to (512, 512)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.head = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, tokens, target_temporal_size: Optional[int] = None):
        """
        Args:
            tokens: (B, T_tokens, L, D) tensor from OCTCube
                    where T_tokens = num_frames / t_patch_size
            target_temporal_size: if provided, upsample temporally to this size

        Returns:
            (B, T_out, num_classes, H, W) segmentation logits
            where T_out = target_temporal_size if provided, else T_tokens
        """
        B, T, L, D = tokens.shape
        h, w = self.patch_grid

        # Handle different spatial dimensions
        expected_L = h * w
        if L != expected_L:
            actual_h = actual_w = int(L ** 0.5)
            h, w = actual_h, actual_w

        # Process all temporal tokens at once
        # (B, T, L, D) -> (B*T, L, D)
        tokens = rearrange(tokens, 'b t l d -> (b t) l d')

        # Project tokens
        x = self.proj(tokens)  # (B*T, L, hidden*64)

        # Reshape: each token becomes 8x8 spatial with hidden channels
        x = rearrange(x, 'bt (h w) (ph pw c) -> bt c (h ph) (w pw)',
                      h=h, w=w, ph=8, pw=8, c=self.hidden_dim)

        # Apply ConvNeXt blocks
        x = self.blocks(x)

        # Upsample spatially and predict
        x = self.upsample(x)
        x = self.head(x)

        # Reshape back: (B*T, C, H, W) -> (B, T, C, H, W)
        _, C, H, W = x.shape
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        # Optionally upsample temporally to match original number of slices
        if target_temporal_size is not None and target_temporal_size != T:
            # (B, T, C, H, W) -> (B, C, T, H, W) for interpolation
            x = x.permute(0, 2, 1, 3, 4)
            x = F.interpolate(x, size=(target_temporal_size, H, W), mode='trilinear', align_corners=False)
            # Back to (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)

        return x


########################################################################
# Complete OCTCube Segmenter


class OCTCubeSegmenter(nn.Module):
    """
    Complete OCTCube segmentation model.

    Combines the OCTCube encoder with a ConvNeXt-based segmentation head
    to produce per-slice segmentations from 3D OCT volumes.

    Default parameters match the OCTCube checkpoint:
    - img_size=512, patch_size=16
    - num_frames=48, t_patch_size=3
    """

    def __init__(
        self,
        num_classes: int = 12,
        img_size: int = 512,
        patch_size: int = 16,
        num_frames: int = 48,
        t_patch_size: int = 3,
        size: str = 'large',
        freeze_encoder: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        # Create encoder
        self.encoder = OCTCubeWrapper(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
            size=size,
        )

        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self.encoder.load_pretrained(checkpoint_path)

        # Freeze encoder if requested
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("Encoder weights frozen")

        # Create segmentation head
        grid_size = img_size // patch_size
        self.head = OCTCubeSegHead(
            num_classes=num_classes,
            in_dim=self.encoder.embed_dim,
            hidden_dim=96,
            patch_grid=(grid_size, grid_size),
        )

        print(f"Created OCTCubeSegmenter with {num_classes} classes, "
              f"img_size={img_size}, num_frames={num_frames}")

    def forward(self, x, return_per_slice: bool = True):
        """
        Args:
            x: (B, T, C, H, W) tensor of OCT volume
               OR (B, T, H, W) which will be unsqueezed
            return_per_slice: if True, upsample temporally to return per-slice predictions

        Returns:
            (B, T, num_classes, H, W) segmentation logits per slice if return_per_slice
            (B, T/t_patch_size, num_classes, H, W) otherwise
        """
        # Handle missing channel dimension
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, T, H, W) -> (B, T, 1, H, W)

        B, T, C, H, W = x.shape

        # Get tokens from encoder
        tokens = self.encoder(x, return_all_tokens=True)  # (B, T/t_patch_size, L, D)

        # Get segmentation with optional temporal upsampling
        target_T = T if return_per_slice else None
        logits = self.head(tokens, target_temporal_size=target_T)

        return logits


########################################################################
# Main function for testing


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model with default OCTCube parameters
    model = OCTCubeSegmenter(
        num_classes=12,
        img_size=512,
        patch_size=16,
        num_frames=48,
        t_patch_size=3,
        size='large',
        freeze_encoder=True,
        checkpoint_path=None,  # Set path to your checkpoint here
    ).to(device)

    print(f"\nModel created on {device}")

    # Test with random input matching OCTCube expectations
    B, T, C, H, W = 1, 48, 1, 512, 512
    x = torch.randn(B, T, C, H, W, device=device)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")
    print(f"Expected: ({B}, {T}, 12, {H}, {W})")
