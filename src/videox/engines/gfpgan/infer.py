"""GFPGAN face restoration in the main environment.

Follows upstream ``GFPGANer.enhance`` (v1.3.8) with ``has_aligned=False`` and
``paste_back=True``: detect five landmarks per face with RetinaFace, warp each
face to 512x512, restore it, upscale the background (Real-ESRGAN or Lanczos)
and paste the faces back with a soft square mask. The face parsing mask that
upstream's helper can use is not ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from videox.engines.gfpgan.arch import GFPGANv1Clean, RetinaFace
from videox.engines.gfpgan.faces import (
    align_face,
    cv2_module,
    detect_faces,
    inverse_affine,
    paste_faces,
)
from videox.engines.gfpgan.presets import (
    BACKGROUND_UPSCALERS,
    BACKGROUND_WEIGHTS_FOLDER,
    DEFAULT_BACKGROUND,
    DEFAULT_FP16,
    DEFAULT_ONLY_CENTER_FACE,
    DEFAULT_STRENGTH,
    DEFAULT_TILE,
    DEFAULT_TILE_PAD,
    DEFAULT_VERSION,
    EYE_DIST_THRESHOLD,
    NO_BACKGROUND,
    GFPGANModel,
    model_spec,
)
from videox.engines.gfpgan.weights import ensure_model
from videox.engines.image_io import (
    StillImage,
    check_output_format,
    load_state_dict,
    pick_device,
    publish_preview,
    read_image,
    resize_plane,
    write_image,
)
from videox.engines.progress import emit_progress
from videox.engines.realesrgan.infer import upscale_rgb
from videox.engines.realesrgan.weights import ensure_model as ensure_background
from videox.errors import VideoXError

UPSCALES = (1, 2, 4)


@dataclass(frozen=True)
class GFPGANRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    version: str = DEFAULT_VERSION
    upscale: int = 2
    background: str = DEFAULT_BACKGROUND
    strength: float = DEFAULT_STRENGTH
    only_center_face: bool = DEFAULT_ONLY_CENTER_FACE
    tile: int = DEFAULT_TILE
    tile_pad: int = DEFAULT_TILE_PAD
    half: bool = DEFAULT_FP16


def background_dir(model_dir: Path) -> Path:
    """Real-ESRGAN weights sit in the sibling folder, shared with the Real-ESRGAN engine."""
    return model_dir.parent / BACKGROUND_WEIGHTS_FOLDER


def run_gfpgan(request: GFPGANRequest) -> None:
    spec = model_spec(request.version)
    _check_request(request)
    check_output_format(request.output_path)
    cv2_module()
    source = read_image(request.input_path)
    device = pick_device()

    emit_progress("download", 0, 1)
    restorer_path, detector_path = ensure_model(request.model_dir, request.version)
    if request.background != NO_BACKGROUND:
        ensure_background(background_dir(request.model_dir), request.background)
    emit_progress("download", 1, 1)

    emit_progress("load", 0, 2)
    detector = load_detector(detector_path, device)
    emit_progress("load", 1, 2)
    restorer = load_restorer(restorer_path, spec, device)
    emit_progress("load", 2, 2)

    rgb = source.rgb
    try:
        restored, inverses = restore_faces(
            rgb,
            detector,
            restorer,
            device=device,
            upscale=request.upscale,
            strength=request.strength,
            only_center_face=request.only_center_face,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise VideoXError("显存不足，人脸修复失败，请换用更小的图片或空闲的显卡") from exc
    del detector, restorer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not restored:
        print("没有检测到人脸，只处理背景。", flush=True)

    background = upscale_background(rgb, request, device)
    output = paste_faces(
        background,
        restored,
        inverses,
        input_size=(source.height, source.width),
        upscale=request.upscale,
    )
    alpha = None
    if source.alpha is not None:
        alpha = resize_plane(source.alpha, int(output.shape[1]), int(output.shape[0]))
    write_image(request.output_path, StillImage(output, alpha))
    publish_preview(source, output)


def load_detector(path: Path, device: torch.device) -> RetinaFace:
    network = RetinaFace()
    try:
        network.load_state_dict(load_state_dict(path), strict=True)
    except RuntimeError as exc:
        raise VideoXError(f"{path.name} 和人脸检测网络对不上，请删除该文件后重试") from exc
    return network.eval().to(device)


def load_restorer(path: Path, spec: GFPGANModel, device: torch.device) -> GFPGANv1Clean:
    network = GFPGANv1Clean(**spec.arch_kwargs)
    try:
        network.load_state_dict(load_state_dict(path), strict=True)
    except RuntimeError as exc:
        raise VideoXError(f"{path.name} 和 GFPGAN {spec.label} 网络对不上，请删除该文件后重试") from exc
    return network.eval().to(device)


def restore_faces(
    rgb: np.ndarray,
    detector: RetinaFace,
    restorer: GFPGANv1Clean,
    *,
    device: torch.device,
    upscale: int,
    strength: float,
    only_center_face: bool,
    randomize_noise: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect, align and restore every face. Returns the restored 512x512 RGB faces and their inverse affines."""
    emit_progress("face", 0, 1)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with torch.inference_mode():
        faces = detect_faces(
            detector,
            bgr,
            only_center_face=only_center_face,
            eye_dist_threshold=EYE_DIST_THRESHOLD,
        )
    total = len(faces)
    if total == 0:
        emit_progress("face", 1, 1)
        return [], []
    emit_progress("face", 0, total)
    # StyleGAN noise is random; a fixed seed makes the same input give the same output.
    torch.manual_seed(0)
    restored: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    for index, face in enumerate(faces):
        matrix, cropped = align_face(rgb, face.landmarks)
        result = restore_face(restorer, cropped, device, randomize_noise=randomize_noise)
        if result is None:
            print(f"第 {index + 1} 张人脸修复失败，保留原样。", flush=True)
            result = cropped
        restored.append(blend_face(result, cropped, strength))
        inverses.append(inverse_affine(matrix, upscale))
        emit_progress("face", index + 1, total)
    return restored, inverses


def restore_face(
    restorer: GFPGANv1Clean,
    cropped: np.ndarray,
    device: torch.device,
    *,
    randomize_noise: bool = True,
) -> np.ndarray | None:
    """Run GFPGAN on one aligned 512x512 RGB face; None when the network fails, like upstream."""
    face = (cropped / 255.0).astype(np.float32)
    tensor = torch.from_numpy(face.transpose(2, 0, 1)).float()
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            output = restorer(tensor, randomize_noise=randomize_noise)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError:
        return None
    image = output.squeeze(0).float().detach().cpu().clamp_(-1, 1)
    image = (image + 1) / 2
    pixels = image.numpy().transpose(1, 2, 0)
    return (pixels * 255.0).round().astype(np.uint8)


def blend_face(restored: np.ndarray, original: np.ndarray, strength: float) -> np.ndarray:
    """Mix the restored face with the aligned original; 1 keeps the restored face as is."""
    if strength >= 1:
        return restored
    mixed = strength * restored.astype(np.float32) + (1 - strength) * original.astype(np.float32)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def upscale_background(rgb: np.ndarray, request: GFPGANRequest, device: torch.device) -> np.ndarray:
    height, width = rgb.shape[:2]
    if request.background == NO_BACKGROUND:
        cv2 = cv2_module()
        emit_progress("restore", 0, 1)
        # Same Lanczos resize upstream's helper uses when there is no background upsampler.
        resized = cv2.resize(
            rgb,
            (int(width * request.upscale), int(height * request.upscale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        emit_progress("restore", 1, 1)
        return resized
    return upscale_rgb(
        rgb,
        model_id=request.background,
        model_dir=background_dir(request.model_dir),
        device=device,
        half=request.half,
        tile=request.tile,
        tile_pad=request.tile_pad,
        outscale=float(request.upscale),
        phase="restore",
    )


def _check_request(request: GFPGANRequest) -> None:
    if request.upscale not in UPSCALES:
        raise VideoXError("放大倍数只能是 1、2 或 4")
    if request.background != NO_BACKGROUND and request.background not in BACKGROUND_UPSCALERS:
        raise VideoXError(f"未知的背景处理方式: {request.background}")
    if not 0 <= request.strength <= 1:
        raise VideoXError("人脸修复强度需要在 0 到 1 之间")
    if request.tile < 0:
        raise VideoXError("背景分块大小不能小于 0")
    if request.tile_pad < 0:
        raise VideoXError("背景分块重叠不能小于 0")
