"""Real-ESRGAN still-image inference in the main environment.

Follows upstream ``inference_realesrgan.py`` (v0.3.0): run the network at its
native scale with ``pre_pad=0``, then Lanczos-resize to ``outscale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from videox.engines.image_io import (
    check_output_size,
    load_state_dict,
    pick_device,
    read_image,
    resize_plane,
    run_tiled,
    save_result,
    scaled_size,
    to_tensor,
)
from videox.engines.progress import emit_progress
from videox.engines.realesrgan.arch import build_network
from videox.engines.realesrgan.presets import (
    DEFAULT_DENOISE_STRENGTH,
    DEFAULT_FP16,
    DEFAULT_MODEL,
    DEFAULT_TILE,
    DEFAULT_TILE_PAD,
    model_spec,
)
from videox.engines.realesrgan.weights import ensure_model
from videox.errors import VideoXError


@dataclass(frozen=True)
class RealESRGANRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    model_id: str = DEFAULT_MODEL
    # None keeps the model's native scale.
    outscale: float | None = None
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH
    tile: int = DEFAULT_TILE
    tile_pad: int = DEFAULT_TILE_PAD
    half: bool = DEFAULT_FP16


def run_realesrgan(request: RealESRGANRequest) -> None:
    spec = model_spec(request.model_id)
    outscale = float(spec.scale if request.outscale is None else request.outscale)
    source = read_image(request.input_path)
    check_output_size(request.output_path, *scaled_size(source.width, source.height, outscale))
    device = pick_device()
    emit_progress("download", 0, 1)
    paths = ensure_model(request.model_dir, request.model_id, denoise_strength=request.denoise_strength)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    model = load_model(
        request.model_id,
        paths,
        device=device,
        half=request.half,
        denoise_strength=request.denoise_strength,
    )
    emit_progress("load", 1, 1)
    rgb = upscale_rgb(
        model,
        source.rgb,
        model_id=request.model_id,
        tile=request.tile,
        tile_pad=request.tile_pad,
        outscale=outscale,
    )
    save_result(request.output_path, source, rgb)


def upscale_rgb(
    model: torch.nn.Module,
    rgb: np.ndarray,
    *,
    model_id: str,
    tile: int,
    tile_pad: int,
    outscale: float,
    phase: str = "restore",
) -> np.ndarray:
    """Run a loaded model over HxWx3 uint8 RGB and return round(w*outscale) x round(h*outscale) RGB."""
    spec = model_spec(model_id)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    parameter = next(model.parameters())
    output = run_tiled(
        model,
        to_tensor(rgb, parameter.device, parameter.dtype),
        scale=spec.scale,
        tile=tile,
        tile_pad=tile_pad,
        multiple_of=spec.multiple_of,
        phase=phase,
    )
    return resize_plane(output, *scaled_size(width, height, outscale))


def load_model(
    model_id: str,
    paths: list[Path],
    *,
    device: torch.device,
    half: bool,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
) -> torch.nn.Module:
    """Build the network and load ``paths`` (a second path is blended in by denoise strength).

    ``half`` only takes effect on CUDA; the CPU always runs in float32.
    """
    spec = model_spec(model_id)
    state = load_state_dict(paths[0])
    if len(paths) > 1:
        state = dni(state, load_state_dict(paths[1]), denoise_strength)
    network = build_network(spec.arch, **spec.arch_kwargs)
    try:
        network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise VideoXError(f"{paths[0].name} 和 {spec.label} 网络对不上，请删除该文件后重试") from exc
    network.eval().to(device)
    if half and device.type == "cuda":
        network.half()
    return network


def dni(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    weight_a: float,
) -> dict[str, torch.Tensor]:
    """Deep network interpolation: ``weight_a * a + (1 - weight_a) * b`` for every parameter."""
    if state_a.keys() != state_b.keys():
        raise VideoXError("两份去噪权重的结构不一致，请删除后重新下载")
    weight_b = 1 - weight_a
    return {key: weight_a * value + weight_b * state_b[key] for key, value in state_a.items()}

