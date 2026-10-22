"""SCUNet network, adapted from cszn/SCUNet (Apache-2.0).

Source: https://github.com/cszn/SCUNet/blob/52e440a80a655b01e0b41e9dd9bfe599bc11625e/models/network_scunet.py
(commit 52e440a on main; the repository has no tags). Swin-Conv-UNet by Kai
Zhang, Yawei Li, Jingyun Liang et al. Copyright the original authors, Apache
License 2.0.

Changes from upstream: inference only (no drop path, init hooks, FLOP counter
or construction prints); einops and timm replaced by plain reshape/permute and
``torch.nn.init.trunc_normal_``; the relative-position lookup is a
non-persistent buffer built once instead of on every forward. State dict keys
are unchanged, so the release checkpoints load with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

HEAD_DIM = 32
WINDOW_SIZE = 8
# The U-Net halves the size three times and the deepest windows are 8 wide.
PAD_MULTIPLE = 64


class WMSA(nn.Module):
    """Window (``W``) or shifted-window (``SW``) multi-head self-attention on (B, H, W, C)."""

    def __init__(self, input_dim: int, output_dim: int, head_dim: int, window_size: int, type: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        span = 2 * window_size - 1
        self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, span, span))
        nn.init.trunc_normal_(self.relative_position_params, std=0.02)
        self.linear = nn.Linear(input_dim, output_dim)
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        relation = coords[:, None, :] - coords[None, :, :] + window_size - 1
        self.register_buffer("relative_rows", relation[..., 0].contiguous(), persistent=False)
        self.register_buffer("relative_cols", relation[..., 1].contiguous(), persistent=False)

    def relative_embedding(self) -> Tensor:
        """(heads, window*window, window*window) bias added to the attention scores."""
        return self.relative_position_params[:, self.relative_rows, self.relative_cols]

    def generate_mask(self, h_windows: int, w_windows: int, shift: int, device: torch.device) -> Tensor:
        """Mask pairs that the cyclic shift brought together from opposite edges (True = blocked)."""
        p = self.window_size
        mask = torch.zeros(h_windows, w_windows, p, p, p, p, dtype=torch.bool, device=device)
        if self.type == "W":
            return mask.view(1, 1, h_windows * w_windows, p * p, p * p)
        s = p - shift
        mask[-1, :, :s, :, s:, :] = True
        mask[-1, :, s:, :, :s, :] = True
        mask[:, -1, :, :s, :, s:] = True
        mask[:, -1, :, s:, :, :s] = True
        return mask.view(1, 1, h_windows * w_windows, p * p, p * p)

    def forward(self, x: Tensor) -> Tensor:
        p = self.window_size
        shift = p // 2
        if self.type != "W":
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        batch, height, width, channels = x.shape
        h_windows, w_windows = height // p, width // p
        windows = h_windows * w_windows
        # b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c
        x = x.view(batch, h_windows, p, w_windows, p, channels).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(batch, windows, p * p, channels)
        qkv = self.embedding_layer(x)
        # b nw np (3h c) -> 3h b nw np c, then q, k, v are consecutive groups of heads.
        qkv = qkv.view(batch, windows, p * p, 3 * self.n_heads, self.head_dim).permute(3, 0, 1, 2, 4)
        q, k, v = qkv.chunk(3, dim=0)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        sim = sim + self.relative_embedding()[:, None, None]
        if self.type != "W":
            sim = sim.masked_fill(self.generate_mask(h_windows, w_windows, shift, x.device), float("-inf"))
        probs = F.softmax(sim, dim=-1)
        output = torch.matmul(probs, v)
        # h b nw np c -> b nw np (h c)
        output = output.permute(1, 2, 3, 0, 4).reshape(batch, windows, p * p, self.n_heads * self.head_dim)
        output = self.linear(output)
        # b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c
        output = output.view(batch, h_windows, w_windows, p, p, self.output_dim).permute(0, 1, 3, 2, 4, 5)
        output = output.reshape(batch, height, width, self.output_dim)
        if self.type != "W":
            output = torch.roll(output, shifts=(shift, shift), dims=(1, 2))
        return output


class Block(nn.Module):
    """Swin Transformer block on (B, H, W, C)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        type: str = "W",
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        if type not in ("W", "SW"):
            raise ValueError(f"unknown attention type: {type}")
        self.type = "W" if input_resolution <= window_size else type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.msa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ConvTransBlock(nn.Module):
    """Split the channels, run a residual conv branch and a Swin branch, then fuse."""

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        type: str = "W",
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        if type not in ("W", "SW"):
            raise ValueError(f"unknown attention type: {type}")
        self.type = "W" if input_resolution <= window_size else type
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, self.type, input_resolution)
        both = conv_dim + trans_dim
        self.conv1_1 = nn.Conv2d(both, both, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(both, both, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


def _stage(dim: int, count: int, resolution: int) -> list[nn.Module]:
    # Blocks alternate plain and shifted windows, starting with plain.
    return [
        ConvTransBlock(dim, dim, HEAD_DIM, WINDOW_SIZE, "W" if not i % 2 else "SW", resolution)
        for i in range(count)
    ]


class SCUNet(nn.Module):
    """Swin-Conv-UNet. Takes and returns (B, in_nc, H, W) in [0, 1]; any H and W."""

    def __init__(
        self,
        in_nc: int = 3,
        config: tuple[int, ...] | list[int] = (2, 2, 2, 2, 2, 2, 2),
        dim: int = 64,
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        if len(config) != 7:
            raise ValueError("config needs seven stage depths")
        self.config = list(config)
        self.dim = dim
        self.head_dim = HEAD_DIM
        self.window_size = WINDOW_SIZE
        res = input_resolution
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *_stage(dim // 2, config[0], res), nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)
        )
        self.m_down2 = nn.Sequential(
            *_stage(dim, config[1], res // 2), nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)
        )
        self.m_down3 = nn.Sequential(
            *_stage(2 * dim, config[2], res // 4), nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)
        )
        self.m_body = nn.Sequential(*_stage(4 * dim, config[3], res // 8))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), *_stage(2 * dim, config[4], res // 4)
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), *_stage(dim, config[5], res // 2)
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), *_stage(dim // 2, config[6], res)
        )
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0: Tensor) -> Tensor:
        height, width = x0.shape[-2:]
        # Replicate-pad bottom and right to a multiple of 64, as upstream does, then crop back.
        pad_h = -height % PAD_MULTIPLE
        pad_w = -width % PAD_MULTIPLE
        if pad_h or pad_w:
            x0 = F.pad(x0, (0, pad_w, 0, pad_h), mode="replicate")
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :height, :width]
