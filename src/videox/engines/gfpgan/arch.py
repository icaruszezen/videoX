"""GFPGAN face restoration network and the RetinaFace (ResNet50) face detector.

GFPGANv1Clean and its StyleGAN2 decoder are adapted from GFPGAN (Apache-2.0,
Copyright (C) 2021 THL A29 Limited, a Tencent company), tag v1.3.8:
https://github.com/TencentARC/GFPGAN/blob/v1.3.8/gfpgan/archs/gfpganv1_clean_arch.py
https://github.com/TencentARC/GFPGAN/blob/v1.3.8/gfpgan/archs/stylegan2_clean_arch.py
with ``default_init_weights`` from BasicSR (Apache-2.0), tag v1.4.2:
https://github.com/XPixelGroup/BasicSR/blob/v1.4.2/basicsr/archs/arch_util.py

RetinaFace is adapted from facexlib (MIT, Copyright (c) 2020 Xintao Wang),
tag v0.3.0:
https://github.com/xinntao/facexlib/blob/v0.3.0/facexlib/detection/retinaface.py
https://github.com/xinntao/facexlib/blob/v0.3.0/facexlib/detection/retinaface_net.py
https://github.com/xinntao/facexlib/blob/v0.3.0/facexlib/detection/retinaface_utils.py
Its ResNet50 backbone is built here with torchvision's layer names (what
``IntermediateLayerGetter(resnet50(), ...)`` keeps), so only the released
detector checkpoint is loaded and nothing is downloaded. ``py_cpu_nms`` called
``torchvision.ops.nms``; ``nms`` below is a numpy port of the same rule.

Removed: registry decorators and BasicSR imports, decoder loading and freezing,
style mixing, truncation and latent outputs, the intermediate RGB outputs
(their weights stay so checkpoints load strictly), the MobileNet backbone,
batched detection, face cropping to 112x112 and training-time box matching.
Parameter names match the released checkpoints.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init


@torch.no_grad()
def default_init_weights(module: nn.Module, scale: float = 1.0, bias_fill: float = 0.0, **kwargs: object) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(layer.weight, **kwargs)
            layer.weight.data *= scale
            if layer.bias is not None:
                layer.bias.data.fill_(bias_fill)


# ---------------------------------------------------------------------------
# StyleGAN2 decoder (clean version, no custom CUDA ops)
# ---------------------------------------------------------------------------


class NormStyleCode(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    """Modulated convolution of StyleGAN2. There is no bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        default_init_weights(self.modulation, scale=1, bias_fill=1, a=0, mode="fan_in", nonlinearity="linear")
        weight = torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight / math.sqrt(in_channels * kernel_size**2))
        self.padding = kernel_size // 2

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, *out.shape[2:4])


class StyleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode
        )
        self.weight = nn.Parameter(torch.zeros(1))  # noise strength
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor, style: Tensor, noise: Tensor | None = None) -> Tensor:
        out = self.modulated_conv(x, style) * 2**0.5
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = out + self.bias
        return self.activate(out)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, num_style_feat: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: Tensor, style: Tensor, skip: Tensor | None = None) -> Tensor:
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel: int, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch: int) -> Tensor:
        return self.weight.repeat(batch, 1, 1, 1)


def _channels(narrow: float, channel_multiplier: int) -> dict[str, int]:
    return {
        "4": int(512 * narrow),
        "8": int(512 * narrow),
        "16": int(512 * narrow),
        "32": int(512 * narrow),
        "64": int(256 * channel_multiplier * narrow),
        "128": int(128 * channel_multiplier * narrow),
        "256": int(64 * channel_multiplier * narrow),
        "512": int(32 * channel_multiplier * narrow),
        "1024": int(16 * channel_multiplier * narrow),
    }


class StyleGAN2GeneratorCSFT(nn.Module):
    """StyleGAN2 generator with SFT modulation (StyleGAN2GeneratorClean + CSFT forward)."""

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1,
        sft_half: bool = False,
    ) -> None:
        super().__init__()
        self.sft_half = sft_half
        self.num_style_feat = num_style_feat
        style_mlp_layers: list[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            style_mlp_layers.extend(
                [nn.Linear(num_style_feat, num_style_feat, bias=True), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            )
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2, mode="fan_in", nonlinearity="leaky_relu")

        channels = _channels(narrow, channel_multiplier)
        self.channels = channels
        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(
            channels["4"],
            channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
        )
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        self.log_size = int(math.log2(out_size))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channels = channels["4"]
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(1, 1, resolution, resolution))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                )
            )
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def forward(
        self,
        styles: Tensor,
        conditions: list[Tensor],
        input_is_latent: bool = False,
        randomize_noise: bool = True,
    ) -> Tensor:
        if not input_is_latent:
            styles = self.style_mlp(styles)
        if randomize_noise:
            noise: list[Tensor | None] = [None] * self.num_layers
        else:
            # The fixed noise maps stored in the checkpoint.
            noise = [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, self.num_latent, 1)
        else:
            # One latent per layer (different_w).
            latent = styles

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            # The conditions may have fewer levels than the decoder.
            if i < len(conditions):
                if self.sft_half:
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        return skip


class ResBlock(nn.Module):
    """Residual block with bilinear upsampling or downsampling."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "down") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.scale_factor = 0.5 if mode == "down" else 2

    def forward(self, x: Tensor) -> Tensor:
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return out + self.skip(x)


class GFPGANv1Clean(nn.Module):
    """U-Net encoder + StyleGAN2 decoder with SFT. Input and output are RGB in [-1, 1]."""

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        channel_multiplier: int = 1,
        num_mlp: int = 8,
        input_is_latent: bool = False,
        different_w: bool = False,
        narrow: float = 1,
        sft_half: bool = False,
    ) -> None:
        super().__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        channels = _channels(narrow * 0.5, channel_multiplier)
        self.log_size = int(math.log2(out_size))
        first_out_size = 2 ** (int(math.log2(out_size)))
        self.conv_body_first = nn.Conv2d(3, channels[f"{first_out_size}"], 1)

        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2 ** (i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels["4"], 3, 1, 1)

        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode="up"))
            in_channels = out_channels

        # Intermediate RGB heads used only in training; kept so checkpoints load strictly.
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f"{2**i}"], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log2(out_size)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat
        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out_channel)

        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half,
        )

        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            sft_out_channels = out_channels if sft_half else out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )

    def forward(self, x: Tensor, randomize_noise: bool = True) -> Tensor:
        conditions: list[Tensor] = []
        unet_skips: list[Tensor] = []
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            conditions.append(self.condition_scale[i](feat))
            conditions.append(self.condition_shift[i](feat))

        return self.stylegan_decoder(
            style_code,
            conditions,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise,
        )


# ---------------------------------------------------------------------------
# RetinaFace (ResNet50)
# ---------------------------------------------------------------------------

MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCE = (0.1, 0.2)
# BGR channel means subtracted from the 0-255 input.
BGR_MEAN = (104.0, 117.0, 123.0)


class Bottleneck(nn.Module):
    """torchvision's ResNet bottleneck (stride on the 3x3 convolution)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _resnet_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * Bottleneck.expansion, 1, stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
    layers = [Bottleneck(inplanes, planes, stride, downsample)]
    layers += [Bottleneck(planes * Bottleneck.expansion, planes) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class ResNet50Body(nn.Module):
    """ResNet50 up to ``layer4``; returns the ``layer2``, ``layer3`` and ``layer4`` features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _resnet_layer(64, 64, 3, 1)
        self.layer2 = _resnet_layer(256, 128, 4, 2)
        self.layer3 = _resnet_layer(512, 256, 6, 2)
        self.layer4 = _resnet_layer(1024, 512, 3, 2)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c2, c3, c4]


def conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


def conv_bn1x1(inp: int, oup: int, stride: int, leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0.1 if out_channel <= 64 else 0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5_1 = self.conv5X5_1(x)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        return F.relu(torch.cat([conv3x3, conv5x5, conv7x7], dim=1))


class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        output1 = self.output1(features[0])
        output2 = self.output2(features[1])
        output3 = self.output3(features[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = self.merge2(output2 + up3)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = self.merge1(output1 + up2)
        return [output1, output2, output3]


class ClassHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    """RetinaFace with a ResNet50 backbone, test phase (softmax scores)."""

    def __init__(self) -> None:
        super().__init__()
        in_channels = 256
        out_channels = 256
        self.body = ResNet50Body()
        self.fpn = FPN([in_channels * 2, in_channels * 4, in_channels * 8], out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = nn.ModuleList([ClassHead(out_channels, 2) for _ in range(3)])
        self.BboxHead = nn.ModuleList([BboxHead(out_channels, 2) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([LandmarkHead(out_channels, 2) for _ in range(3)])
        self.register_buffer("mean", torch.tensor(BGR_MEAN).view(1, 3, 1, 1), persistent=False)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``inputs`` is BGR, 0-255, mean already subtracted."""
        fpn = self.fpn(self.body(inputs))
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions

    @torch.no_grad()
    def detect_faces(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.8,
        nms_threshold: float = 0.4,
    ) -> np.ndarray:
        """Detect faces in an HxWx3 BGR image at its own size (``use_origin_size=True``).

        Returns an Nx15 float32 array: box x1, y1, x2, y2, score, then five
        landmarks as x, y pairs (eyes, nose, mouth corners), best score first.
        """
        device = self.mean.device
        height, width = image.shape[:2]
        inputs = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(device)
        inputs = inputs - self.mean
        loc, conf, landmarks = self(inputs)
        priors = prior_box(height, width).to(device)
        scale = torch.tensor([width, height] * 2, dtype=torch.float32, device=device)
        scale1 = torch.tensor([width, height] * 5, dtype=torch.float32, device=device)

        boxes = (decode(loc.squeeze(0), priors, VARIANCE) * scale).cpu().numpy()
        scores = conf.squeeze(0).cpu().numpy()[:, 1]
        points = (decode_landm(landmarks.squeeze(0), priors, VARIANCE) * scale1).cpu().numpy()

        inds = np.where(scores > conf_threshold)[0]
        boxes, points, scores = boxes[inds], points[inds], scores[inds]
        order = scores.argsort()[::-1]
        boxes, points, scores = boxes[order], points[order], scores[order]
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(bounding_boxes, nms_threshold)
        return np.concatenate((bounding_boxes[keep, :], points[keep]), axis=1)


def prior_box(height: int, width: int) -> Tensor:
    """Anchor centres and sizes, normalised to the image, in facexlib's PriorBox order."""
    anchors = []
    for step, min_sizes in zip(STEPS, MIN_SIZES):
        rows, cols = math.ceil(height / step), math.ceil(width / step)
        cy = (np.arange(rows, dtype=np.float64) + 0.5) * step / height
        cx = (np.arange(cols, dtype=np.float64) + 0.5) * step / width
        grid_y, grid_x = np.meshgrid(cy, cx, indexing="ij")
        per_size = [
            np.stack(
                [grid_x, grid_y, np.full_like(grid_x, size / width), np.full_like(grid_x, size / height)],
                axis=-1,
            )
            for size in min_sizes
        ]
        # Row, then column, then anchor size, like the nested loops upstream.
        anchors.append(np.stack(per_size, axis=2).reshape(-1, 4))
    return torch.from_numpy(np.concatenate(anchors, axis=0)).float()


def decode(loc: Tensor, priors: Tensor, variances: tuple[float, float]) -> Tensor:
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: Tensor, priors: Tensor, variances: tuple[float, float]) -> Tensor:
    return torch.cat(
        [priors[:, :2] + pre[:, 2 * k : 2 * k + 2] * variances[0] * priors[:, 2:] for k in range(5)],
        dim=1,
    )


def nms(dets: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy NMS with ``torchvision.ops.nms`` rules: drop boxes whose IoU with a kept one is above ``threshold``."""
    if len(dets) == 0:
        return np.zeros(0, dtype=np.int64)
    boxes = dets[:, :4].astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-dets[:, 4].astype(np.float32), kind="stable")
    suppressed = np.zeros(len(dets), dtype=bool)
    keep: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        keep.append(int(index))
        width = np.maximum(np.minimum(x2[index], x2) - np.maximum(x1[index], x1), 0)
        height = np.maximum(np.minimum(y2[index], y2) - np.maximum(y1[index], y1), 0)
        inter = width * height
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = inter / (areas[index] + areas - inter)
        suppressed |= iou > threshold
    return np.asarray(keep, dtype=np.int64)
