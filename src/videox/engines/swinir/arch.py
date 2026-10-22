"""SwinIR network, adapted from JingyunLiang/SwinIR (Apache-2.0).

Source: https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py
(commit 6545850 on main; tag v0.0 predates the x2 real-photo and colour JPEG
settings). Originally written by Ze Liu for Swin Transformer, modified by
Jingyun Liang. Copyright the original authors, Apache License 2.0.

Changes from upstream: inference only (no dropout, drop path, gradient
checkpointing, absolute position embedding, patch merging or FLOP counters);
torch-only helpers instead of timm; only the ``nearest+conv`` and plain
residual heads used by the real-photo, denoising and JPEG checkpoints; the RGB
mean is a non-persistent buffer; the shifted-window mask for a non-training
size is built once per forward instead of once per block. State dict keys are
unchanged, including the ``attn_mask`` and ``relative_position_index`` buffers
that the release checkpoints store.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RGB_MEAN = (0.4488, 0.4371, 0.4040)


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) -> (B * windows, window, window, C)."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    """(B * windows, window, window, C) -> (B, H, W, C)."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def shifted_window_mask(height: int, width: int, window_size: int, shift_size: int) -> Tensor:
    """Mask for shifted windows: 0 within one region, -100 between regions."""
    img_mask = torch.zeros((1, height, width, 1))
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for h in slices:
        for w in slices:
            img_mask[:, h, w, :] = count
            count += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Window multi-head self attention with a relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(tokens, tokens, -1)
        # In-place adds keep one fewer copy of the (windows, heads, N, N) matrix alive.
        attn.add_(bias.permute(2, 0, 1).contiguous().unsqueeze(0))
        if mask is not None:
            windows = mask.shape[0]
            attn.view(batch_windows // windows, windows, self.num_heads, tokens, tokens).add_(
                mask.unsqueeze(1).unsqueeze(0)
            )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        attn_mask = shifted_window_mask(*input_resolution, window_size, shift_size) if shift_size > 0 else None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor, x_size: tuple[int, int], shift_mask: Tensor | None) -> Tensor:
        height, width = x_size
        batch, _tokens, channels = x.shape
        size = self.window_size
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # The stored mask covers the training size; other sizes use the one built per forward.
            mask = self.attn_mask if x_size == self.input_resolution else shift_mask
        else:
            mask = None
        windows = window_partition(x, size).view(-1, size * size, channels)
        windows = self.attn(windows, mask=mask).view(-1, size, size, channels)
        x = window_reverse(windows, size, height, width)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(batch, height * width, channels)
        return x + self.mlp(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dim,
                input_resolution,
                num_heads,
                window_size,
                0 if index % 2 == 0 else window_size // 2,
                mlp_ratio,
            )
            for index in range(depth)
        )

    def forward(self, x: Tensor, x_size: tuple[int, int], shift_mask: Tensor | None) -> Tensor:
        for block in self.blocks:
            x = block(x, x_size, shift_mask)
        return x


def _residual_conv(dim: int, kind: str) -> nn.Module:
    if kind == "1conv":
        return nn.Conv2d(dim, dim, 3, 1, 1)
    if kind == "3conv":
        # Fewer parameters for the large model.
        return nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    raise ValueError(f"unsupported resi_connection: {kind}")


def _to_tokens(x: Tensor) -> Tensor:
    return x.flatten(2).transpose(1, 2)


def _to_image(x: Tensor, x_size: tuple[int, int]) -> Tensor:
    batch, _tokens, channels = x.shape
    return x.transpose(1, 2).view(batch, channels, x_size[0], x_size[1])


class RSTB(nn.Module):
    """Residual Swin Transformer block: a group of Swin blocks, a conv and a skip."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        resi_connection: str,
    ) -> None:
        super().__init__()
        self.residual_group = BasicLayer(dim, input_resolution, depth, num_heads, window_size, mlp_ratio)
        self.conv = _residual_conv(dim, resi_connection)

    def forward(self, x: Tensor, x_size: tuple[int, int], shift_mask: Tensor | None) -> Tensor:
        return _to_tokens(self.conv(_to_image(self.residual_group(x, x_size, shift_mask), x_size))) + x


class PatchEmbed(nn.Module):
    """Flatten to tokens (patch size 1) and normalise."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(_to_tokens(x))


class SwinIR(nn.Module):
    """SwinIR for real-photo super-resolution (``nearest+conv``), denoising and JPEG cleanup (``""``).

    Input and output are RGB in [0, 1]; ``img_range`` scales internally.
    """

    def __init__(
        self,
        *,
        upscale: int = 4,
        in_chans: int = 3,
        img_size: int = 64,
        window_size: int = 8,
        img_range: float = 1.0,
        depths: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        embed_dim: int = 180,
        num_heads: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        upsampler: str = "nearest+conv",
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        if upsampler == "nearest+conv" and upscale not in (2, 4):
            raise ValueError("nearest+conv supports x2 and x4")
        if upsampler == "" and upscale != 1:
            raise ValueError("the residual head does not upscale")
        if upsampler not in ("nearest+conv", ""):
            raise ValueError(f"unsupported upsampler: {upsampler}")
        if img_size <= window_size:
            raise ValueError("img_size must be larger than window_size")
        num_feat = 64
        self.img_range = img_range
        mean = RGB_MEAN if in_chans == 3 else (0.0,) * in_chans
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, in_chans, 1, 1), persistent=False)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.patches_resolution = (img_size, img_size)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim)
        self.layers = nn.ModuleList(
            RSTB(
                embed_dim,
                self.patches_resolution,
                depth,
                heads,
                window_size,
                mlp_ratio,
                resi_connection,
            )
            for depth, heads in zip(depths, num_heads)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = _residual_conv(embed_dim, resi_connection)

        if upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(_init_weights)

    def check_image_size(self, x: Tensor) -> Tensor:
        _batch, _channels, height, width = x.size()
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        return F.pad(x, (0, pad_w, 0, pad_h), "reflect")

    def forward_features(self, x: Tensor) -> Tensor:
        x_size = (int(x.shape[2]), int(x.shape[3]))
        shift_mask = None
        if x_size != self.patches_resolution:
            shift_mask = shifted_window_mask(*x_size, self.window_size, self.window_size // 2).to(x.device)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size, shift_mask)
        return _to_image(self.norm(x), x_size)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[2:]
        x = self.check_image_size(x)
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        if self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            first = self.conv_first(x)
            residual = self.conv_after_body(self.forward_features(first)) + first
            x = x + self.conv_last(residual)
        x = x / self.img_range + mean
        return x[:, :, : height * self.upscale, : width * self.upscale]


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
