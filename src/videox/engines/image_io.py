"""Image reading, tiled inference and previews for the still-image engines.

This module imports torch, numpy and Pillow, so only the engine subprocesses
use it. The API server never imports it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from videox.engines.preview_io import publish_pair
from videox.engines.progress import emit_progress
from videox.errors import VideoXError

PREVIEW_LONG_EDGE = 2560
_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
_ALPHA_FORMATS = {"PNG", "WEBP", "TIFF"}


@dataclass
class StillImage:
    """An 8-bit RGB image and, when it has real transparency, its alpha plane."""

    rgb: np.ndarray
    alpha: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise VideoXError("主环境缺少 Pillow，请运行 videox setup") from exc
    Image.MAX_IMAGE_PIXELS = None
    return Image, ImageOps


def read_image(path: Path) -> StillImage:
    """Read an image upright (EXIF rotation applied) as 8-bit RGB plus optional alpha."""
    Image, ImageOps = _pillow()
    if not path.is_file():
        raise VideoXError(f"输入文件不存在: {path}")
    try:
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
    except OSError as exc:
        raise VideoXError(f"无法读取图片 {path.name}: {exc}") from exc
    if image.mode in {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}:
        image = Image.fromarray(_to_uint8(np.asarray(image)))
    has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = np.asarray(image.convert("RGBA"))
        rgb = np.array(rgba[..., :3])
        alpha = np.array(rgba[..., 3])
        if int(alpha.min()) == 255:
            alpha = None
        return StillImage(rgb, alpha)
    # np.array copies: Pillow's own buffer is read-only and torch warns about those.
    return StillImage(np.array(image.convert("RGB")))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    data = array.astype(np.float64)
    peak = float(data.max()) if data.size else 0.0
    if peak > 255:
        data = data * (255.0 / (65535.0 if peak <= 65535 else peak))
    elif 0 < peak <= 1.0 and array.dtype.kind == "f":
        data = data * 255.0
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def check_output_format(path: Path) -> str:
    """Fail before any inference when the output suffix cannot be written."""
    fmt = _FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise VideoXError(f"不支持的图片格式: {path.suffix or path.name}")
    return fmt


def write_image(path: Path, image: StillImage) -> None:
    """Write by suffix. PNG, WebP and TIFF keep alpha; JPEG and BMP drop it."""
    Image, _ImageOps = _pillow()
    fmt = check_output_format(path)
    rgb = np.ascontiguousarray(image.rgb, dtype=np.uint8)
    if image.alpha is not None and fmt in _ALPHA_FORMATS:
        pixels = np.dstack([rgb, np.ascontiguousarray(image.alpha, dtype=np.uint8)])
        picture = Image.fromarray(pixels)
    else:
        picture = Image.fromarray(rgb)
    options: dict = {}
    if fmt == "JPEG":
        options = {"quality": 95, "subsampling": 0}
    elif fmt == "WEBP":
        options = {"quality": 95, "method": 4}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        picture.save(partial, format=fmt, **options)
        partial.replace(path)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise VideoXError(f"写出图片失败 {path.name}: {exc}") from exc


def resize_plane(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Lanczos resize of an RGB image or a single alpha plane."""
    Image, _ImageOps = _pillow()
    if array.shape[1] == width and array.shape[0] == height:
        return array
    resized = Image.fromarray(np.ascontiguousarray(array, dtype=np.uint8)).resize(
        (max(int(width), 1), max(int(height), 1)), Image.Resampling.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(resized))


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)


def pick_device() -> torch.device:
    """The progress entry has already narrowed CUDA_VISIBLE_DEVICES to one card."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("没有可用的 CUDA 显卡，改用 CPU，速度会慢很多。", flush=True)
    return torch.device("cpu")


def load_state_dict(path: Path, keys: tuple[str, ...] = ("params_ema", "params")) -> dict[str, torch.Tensor]:
    """Load a checkpoint and unwrap the usual ``params_ema`` / ``module.`` layers."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise VideoXError(f"权重文件损坏，请删除后重试: {path}") from exc
    if isinstance(state, dict):
        for key in keys:
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise VideoXError(f"权重格式无法识别: {path.name}")
    return {
        (name[len("module.") :] if name.startswith("module.") else name): value
        for name, value in state.items()
    }


def to_tensor(rgb: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3xHxW in [0, 1]."""
    pixels = rgb if rgb.flags.writeable and rgb.flags.c_contiguous else np.array(rgb)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype).div_(255)


def to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """1x3xHxW in [0, 1] -> HxWx3 uint8."""
    image = tensor.detach()[0].float().clamp(0, 1).mul(255).round().byte()
    return np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())


def forward_padded(
    model: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    scale: int,
    multiple_of: int = 1,
) -> torch.Tensor:
    """Pad bottom and right to ``multiple_of``, run, then crop to ``scale`` times the input."""
    _batch, _channels, height, width = image.shape
    pad_h = (multiple_of - height % multiple_of) % multiple_of
    pad_w = (multiple_of - width % multiple_of) % multiple_of
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    output = model(image)
    return output[:, :, : height * scale, : width * scale]


def run_tiled(
    model: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    *,
    scale: int,
    tile: int,
    tile_pad: int,
    multiple_of: int = 1,
    phase: str = "restore",
) -> np.ndarray:
    """Run ``model`` over ``image`` (1x3xHxW on the model device) and return HxWx3 uint8.

    ``tile`` <= 0 runs the whole image at once. Otherwise the image is split
    into ``tile``-sized blocks, each read with ``tile_pad`` pixels of context on
    every side; only the centre of each result is kept, so seams stay hidden.
    The output is assembled on the CPU to keep large results out of VRAM.
    """
    _batch, _channels, height, width = image.shape
    if tile <= 0 or (height <= tile and width <= tile):
        emit_progress(phase, 0, 1)
        with torch.inference_mode():
            result = to_rgb(forward_padded(model, image, scale, multiple_of))
        emit_progress(phase, 1, 1)
        return result
    tiles_x = math.ceil(width / tile)
    tiles_y = math.ceil(height / tile)
    total = tiles_x * tiles_y
    output = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
    emit_progress(phase, 0, total)
    done = 0
    for row in range(tiles_y):
        for col in range(tiles_x):
            x0, y0 = col * tile, row * tile
            x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
            px0, py0 = max(x0 - tile_pad, 0), max(y0 - tile_pad, 0)
            px1, py1 = min(x1 + tile_pad, width), min(y1 + tile_pad, height)
            with torch.inference_mode():
                patch = forward_padded(model, image[:, :, py0:py1, px0:px1], scale, multiple_of)
            top, left = (y0 - py0) * scale, (x0 - px0) * scale
            kept = patch[:, :, top : top + (y1 - y0) * scale, left : left + (x1 - x0) * scale]
            output[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = to_rgb(kept)
            done += 1
            emit_progress(phase, done, total)
    return output


def publish_preview(source: StillImage, output_rgb: np.ndarray) -> None:
    """Write the before/after pair the desktop compare view reads, as frame 1."""
    directory = os.environ.get("VIDEOX_PREVIEW_DIR")
    if not directory:
        return
    height, width = output_rgb.shape[:2]
    longest = max(width, height)
    shown = output_rgb
    if longest > PREVIEW_LONG_EDGE:
        ratio = PREVIEW_LONG_EDGE / longest
        shown = resize_plane(output_rgb, *scaled_size(width, height, ratio))
    shown = np.ascontiguousarray(shown)
    publish_pair(
        directory,
        1,
        shown.tobytes(),
        int(shown.shape[1]),
        int(shown.shape[0]),
        np.ascontiguousarray(source.rgb).tobytes(),
        source.width,
        source.height,
    )
