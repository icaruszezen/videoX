"""SCUNet still-image denoising in the main environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from videox.engines.image_io import (
    check_output_size,
    load_state_dict,
    pick_device,
    read_image,
    run_tiled,
    save_result,
    to_tensor,
)
from videox.engines.progress import emit_progress
from videox.engines.scunet.arch import SCUNet
from videox.engines.scunet.presets import SCUNetModel, model_spec
from videox.engines.scunet.weights import ensure_model
from videox.errors import VideoXError


@dataclass(frozen=True)
class SCUNetRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    model: str
    strength: float
    tile: int
    tile_pad: int


def run_scunet(request: SCUNetRequest) -> None:
    spec = model_spec(request.model)
    source = read_image(request.input_path)
    check_output_size(request.output_path, source.width, source.height)
    emit_progress("download", 0, 1)
    (checkpoint,) = ensure_model(request.model_dir, spec.id)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    device = pick_device()
    model = load_model(checkpoint, spec, device)
    emit_progress("load", 1, 1)
    # SCUNet pads every tile to a multiple of 64 itself, so run_tiled does not pad.
    restored = run_tiled(
        blended(model, request.strength),
        to_tensor(source.rgb, device),
        scale=spec.scale,
        tile=request.tile,
        tile_pad=request.tile_pad,
    )
    save_result(request.output_path, source, restored)


def build_model(spec: SCUNetModel) -> SCUNet:
    return SCUNet(**spec.arch_kwargs())


def load_model(path: Path, spec: SCUNetModel, device: torch.device) -> SCUNet:
    model = build_model(spec)
    # The release files are plain state dicts, loaded as upstream does.
    state = load_state_dict(path, keys=())
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise VideoXError(f"权重和模型不匹配，请删除后重试: {path.name}") from exc
    return model.eval().to(device)


def blended(model: SCUNet, strength: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Mix the denoised result back with the input: strength * denoised + (1 - strength) * input."""
    if strength >= 1:
        return model

    def forward(image: torch.Tensor) -> torch.Tensor:
        restored = model(image).clamp_(0, 1)
        return restored.mul_(strength).add_(image, alpha=1 - strength)

    return forward
