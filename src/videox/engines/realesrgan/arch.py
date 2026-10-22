"""Real-ESRGAN networks: RRDBNet and SRVGGNetCompact.

RRDBNet is adapted from BasicSR (Apache-2.0), tag v1.4.2:
https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/rrdbnet_arch.py
with ``default_init_weights``, ``make_layer`` and ``pixel_unshuffle`` from
https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/arch_util.py

SRVGGNetCompact is adapted from Real-ESRGAN (BSD-3-Clause, Copyright (c) 2021,
Xintao Wang), tag v0.3.0:
https://github.com/xinntao/Real-ESRGAN/blob/v0.3.0/realesrgan/archs/srvgg_arch.py

The registry decorators and the BasicSR imports are removed. Parameter names
match the released ``params_ema`` / ``params`` checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from videox.errors import VideoXError


def default_init_weights(modules: list[nn.Module], scale: float = 1.0, bias_fill: float = 0.0) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight)
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)


def make_layer(block: type[nn.Module], count: int, **kwargs: int) -> nn.Sequential:
    return nn.Sequential(*[block(**kwargs) for _ in range(count)])


def pixel_unshuffle(x: Tensor, scale: int) -> Tensor:
    batch, channels, height, width = x.size()
    if height % scale or width % scale:
        raise ValueError("pixel_unshuffle needs height and width divisible by scale")
    h, w = height // scale, width // scale
    view = x.reshape(batch, channels, h, scale, w, scale)
    return view.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * scale * scale, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator. Scales 2 and 1 pixel-unshuffle the input first."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"unknown activation: {act_type}")


class SRVGGNetCompact(nn.Module):
    """Plain conv stack in LR space, one pixel shuffle, plus a nearest-upsampled residual."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(_activation(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(_activation(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def build_network(arch: str, **kwargs: int | str) -> nn.Module:
    if arch == "rrdbnet":
        return RRDBNet(**kwargs)  # type: ignore[arg-type]
    if arch == "srvgg":
        return SRVGGNetCompact(**kwargs)  # type: ignore[arg-type]
    raise VideoXError(f"未知的 Real-ESRGAN 网络结构: {arch}")
