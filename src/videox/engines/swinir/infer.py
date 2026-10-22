"""SwinIR still-image inference in the main environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from videox.engines.image_io import (
    StillImage,
    check_output_format,
    load_state_dict,
    pick_device,
    publish_preview,
    read_image,
    resize_plane,
    run_tiled,
    scaled_size,
    to_tensor,
    write_image,
)
from videox.engines.progress import emit_progress
from videox.engines.swinir.arch import SwinIR
from videox.engines.swinir.presets import MAX_OUTSCALE, SwinIRModel, model_spec
from videox.engines.swinir.weights import ensure_model
from videox.errors import VideoXError


@dataclass(frozen=True)
class SwinIRRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    model: str
    outscale: float | None
    tile: int
    tile_pad: int


def run_swinir(request: SwinIRRequest) -> None:
    spec = model_spec(request.model)
    if request.tile < 0 or request.tile_pad < 0:
        raise VideoXError("分块大小和分块重叠不能是负数")
    outscale = float(request.outscale) if request.outscale is not None else float(spec.scale)
    if not 1 <= outscale <= MAX_OUTSCALE:
        raise VideoXError(f"输出倍数需要在 1 到 {MAX_OUTSCALE} 之间")
    check_output_format(request.output_path)
    source = read_image(request.input_path)
    emit_progress("download", 0, 1)
    (checkpoint,) = ensure_model(request.model_dir, spec.id)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    device = pick_device()
    model = load_model(checkpoint, spec, device)
    emit_progress("load", 1, 1)
    try:
        restored = run_tiled(
            mirror_padded(model, spec.window_size),
            to_tensor(source.rgb, device),
            scale=spec.scale,
            tile=request.tile,
            tile_pad=request.tile_pad,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise VideoXError("显存不足，请调小分块大小后重试") from exc
    width, height = scaled_size(source.width, source.height, outscale)
    restored = resize_plane(restored, width, height)
    alpha = resize_plane(source.alpha, width, height) if source.alpha is not None else None
    write_image(request.output_path, StillImage(restored, alpha))
    publish_preview(source, restored)


def build_model(spec: SwinIRModel) -> SwinIR:
    return SwinIR(**spec.arch_kwargs())


def load_model(path: Path, spec: SwinIRModel, device: torch.device) -> SwinIR:
    model = build_model(spec)
    # Same lookup as upstream define_model: the model's key if present, else the whole file.
    state = load_state_dict(path, keys=(spec.param_key,))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise VideoXError(f"权重和模型不匹配，请删除后重试: {path.name}") from exc
    return model.eval().to(device)


def mirror_pad(image: torch.Tensor, window_size: int) -> torch.Tensor:
    """Pad bottom and right by mirroring, as upstream main_test_swinir.py does.

    Upstream always adds 1 to ``window_size`` rows and columns, reaching the
    next multiple above the size. Tiny inputs are mirrored more than once.
    """
    height, width = image.shape[2:]
    target_h = (height // window_size + 1) * window_size
    target_w = (width // window_size + 1) * window_size
    while image.shape[2] < target_h:
        image = torch.cat([image, torch.flip(image, [2])], 2)
    image = image[:, :, :target_h, :]
    while image.shape[3] < target_w:
        image = torch.cat([image, torch.flip(image, [3])], 3)
    return image[:, :, :, :target_w]


def mirror_padded(model: SwinIR, window_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap ``model`` so every whole image or tile is padded and cropped like upstream."""

    def forward(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[2:]
        output = model(mirror_pad(image, window_size))
        return output[:, :, : height * model.upscale, : width * model.upscale]

    return forward
