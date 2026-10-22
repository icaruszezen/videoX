"""RealBasicVSR x4 network, adapted from OpenMMLab MMagic (Apache-2.0).

The checkpoint stores ``generator.*`` weights. SPyNet convolutions keep the
``conv.weight`` names used by mmcv's ConvModule.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from videox.errors import VideoXError


def flow_warp(
    x: Tensor,
    flow: Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    _batch, _channels, height, width = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=x.dtype),
        torch.arange(width, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).expand(x.size(0), -1, -1, -1)
    grid = grid + flow
    grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sample_grid = torch.stack((grid_x, grid_y), dim=3)
    return F.grid_sample(
        x,
        sample_grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        *,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.activate = nn.ReLU(inplace=True) if activate else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        blocks = [ResidualBlockNoBN(mid_channels=out_channels) for _ in range(num_blocks)]
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*blocks),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale_factor)


class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (8, 32, 64, 32, 16, 2)
        layers = []
        for index in range(5):
            layers.append(
                ConvModule(
                    channels[index],
                    channels[index + 1],
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    activate=index < 4,
                )
            )
        self.basic_module = nn.Sequential(*layers)

    def forward(self, tensor_input: Tensor) -> Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: Tensor, supp: Tensor) -> Tensor:
        batch, _channels, height, width = ref.size()
        ref_pyramid = [(ref - self.mean) / self.std]
        supp_pyramid = [(supp - self.mean) / self.std]
        for _level in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]
        flow = ref.new_zeros(batch, 2, height // 32, width // 32)
        for level, module in enumerate(self.basic_module):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supp_pyramid[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + module(torch.cat([ref_pyramid[level], warped, flow_up], dim=1))
        return flow

    def forward(self, ref: Tensor, supp: Tensor) -> Tensor:
        _batch, _channels, height, width = ref.shape
        width_up = width if width % 32 == 0 else 32 * (width // 32 + 1)
        height_up = height if height % 32 == 0 else 32 * (height // 32 + 1)
        ref_up = F.interpolate(ref, size=(height_up, width_up), mode="bilinear", align_corners=False)
        supp_up = F.interpolate(supp, size=(height_up, width_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref_up, supp_up), size=(height, width), mode="bilinear", align_corners=False)
        flow = flow.clone()
        flow[:, 0] *= float(width) / float(width_up)
        flow[:, 1] *= float(height) / float(height_up)
        return flow


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 20) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lrs: Tensor) -> Tensor:
        batch, time, _channels, height, width = lrs.size()
        flows_forward, flows_backward = self._compute_flow(lrs)
        outputs: list[Tensor] = []
        feat_prop = lrs.new_zeros(batch, self.mid_channels, height, width)
        for index in range(time - 1, -1, -1):
            if index < time - 1:
                flow = flows_backward[:, index]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = self.backward_resblocks(torch.cat([lrs[:, index], feat_prop], dim=1))
            outputs.append(feat_prop)
        outputs.reverse()
        feat_prop = torch.zeros_like(feat_prop)
        for index in range(time):
            lr_curr = lrs[:, index]
            if index > 0:
                flow = flows_forward[:, index - 1] if flows_forward is not None else flows_backward[:, -index]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = self.forward_resblocks(torch.cat([lr_curr, feat_prop], dim=1))
            out = self.lrelu(self.fusion(torch.cat([outputs[index], feat_prop], dim=1)))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out = out + self.img_upsample(lr_curr)
            outputs[index] = out
        return torch.stack(outputs, dim=1)

    def _compute_flow(self, lrs: Tensor) -> tuple[Tensor | None, Tensor]:
        batch, time, channels, height, width = lrs.size()
        if time < 2:
            return None, lrs.new_zeros(batch, 0, 2, height, width)
        first = lrs[:, :-1].reshape(-1, channels, height, width)
        second = lrs[:, 1:].reshape(-1, channels, height, width)
        flows_backward = self.spynet(first, second).view(batch, time - 1, 2, height, width)
        mirror = False
        if time % 2 == 0:
            left, right = torch.chunk(lrs, 2, dim=1)
            mirror = float(torch.norm(left - right.flip(1))) == 0.0
        flows_forward = None if mirror else self.spynet(second, first).view(batch, time - 1, 2, height, width)
        return flows_forward, flows_backward


class RealBasicVSRNet(nn.Module):
    def __init__(
        self,
        mid_channels: int = 64,
        num_propagation_blocks: int = 20,
        num_cleaning_blocks: int = 20,
        dynamic_refine_thres: float = 255,
    ) -> None:
        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.0
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lqs: Tensor) -> Tensor:
        _batch, _time, channels, height, width = lqs.size()
        for _attempt in range(3):
            flat = lqs.reshape(-1, channels, height, width)
            residues = self.image_cleaning(flat)
            lqs = (flat + residues).view(lqs.shape)
            if torch.mean(torch.abs(residues)) < self.dynamic_refine_thres:
                break
        return self.basicvsr(lqs)


def load_realbasicvsr(path: Path, device: torch.device) -> RealBasicVSRNet:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise VideoXError(f"无法读取 RealBasicVSR 权重: {exc}") from exc
    state = raw["state_dict"] if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict) else raw
    if not isinstance(state, dict):
        raise VideoXError("RealBasicVSR 权重格式无法识别")
    weights = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if not key.startswith("generator.") or key.startswith("generator_ema"):
            continue
        weights[key.removeprefix("generator.")] = value
    model = RealBasicVSRNet()
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        missing_text = ", ".join(missing[:8])
        unexpected_text = ", ".join(unexpected[:8])
        raise VideoXError(
            "RealBasicVSR 权重和网络对不上。"
            f" 缺少 {len(missing)} 项（{missing_text}）。"
            f" 多余 {len(unexpected)} 项（{unexpected_text}）。"
        )
    model.to(device)
    model.eval()
    return model
