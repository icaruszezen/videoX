"""GFPGAN face restoration in the main environment.

Follows upstream ``GFPGANer.enhance`` (v1.3.8) with ``has_aligned=False`` and
``paste_back=True``: detect five landmarks per face with RetinaFace, warp each
face to 512x512, restore it, upscale the background (Real-ESRGAN or Lanczos)
and paste the faces back with a soft square mask. Upstream ``GFPGANer`` always
pastes with a face parsing mask instead; that model is not ported, so face
edges blend over a slightly larger square area than upstream.
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
    check_output_size,
    load_state_dict,
    pick_device,
    read_image,
    save_result,
)
from videox.engines.progress import emit_progress
from videox.engines.realesrgan.infer import load_model as load_background_model
from videox.engines.realesrgan.infer import upscale_rgb
from videox.engines.realesrgan.weights import ensure_model as ensure_background
from videox.errors import VideoXError


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
    cv2 = cv2_module()
    source = read_image(request.input_path)
    check_output_size(
        request.output_path,
        int(source.width * request.upscale),
        int(source.height * request.upscale),
    )
    device = pick_device()
    # Strength 0 keeps the background everywhere, so the face networks are not needed.
    fix_faces = request.strength > 0

    emit_progress("download", 0, 1)
    if fix_faces:
        restorer_path, detector_path = ensure_model(request.model_dir, request.version)
    background_paths: list[Path] = []
    if request.background != NO_BACKGROUND:
        background_paths = ensure_background(background_dir(request.model_dir), request.background)
    emit_progress("download", 1, 1)

    restored: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    if fix_faces:
        emit_progress("load", 0, 2)
        detector = load_detector(detector_path, device)
        emit_progress("load", 1, 2)
        restorer = load_restorer(restorer_path, spec, device)
        emit_progress("load", 2, 2)
        try:
            restored, inverses = restore_faces(
                source.rgb,
                detector,
                restorer,
                device=device,
                upscale=request.upscale,
                only_center_face=request.only_center_face,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise VideoXError("显存不足，人脸修复失败，请换用更小的图片或空闲的显卡") from exc
        del detector, restorer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not restored:
            print("没有修复任何人脸，只处理背景。", flush=True)

    try:
        background = upscale_background(source.rgb, request, background_paths, device)
        output = paste_faces(
            background,
            restored,
            inverses,
            input_size=(source.height, source.width),
            upscale=request.upscale,
            strength=request.strength,
        )
    except (MemoryError, cv2.error) as exc:
        if not isinstance(exc, MemoryError) and "Insufficient memory" not in str(exc):
            raise
        raise VideoXError("内存不足，请换用更小的图片或更低的放大倍数") from exc
    save_result(request.output_path, source, output)


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
    only_center_face: bool,
    randomize_noise: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect, align and restore every face. Returns the restored 512x512 RGB faces and their inverse affines.

    A face the network fails on is left out, so the background shows there.
    """
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
    # StyleGAN noise is random; a fixed seed makes the same input give the same output.
    torch.manual_seed(0)
    restored: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    for index, face in enumerate(faces):
        matrix, cropped = align_face(rgb, face.landmarks)
        try:
            result = restore_face(restorer, cropped, device, randomize_noise=randomize_noise)
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            # Upstream also keeps going and prints the error.
            print(f"第 {index + 1} 张人脸修复失败，保留原样: {exc}", flush=True)
        else:
            restored.append(result)
            inverses.append(inverse_affine(matrix, upscale))
        emit_progress("face", index + 1, total)
    return restored, inverses


def restore_face(
    restorer: GFPGANv1Clean,
    cropped: np.ndarray,
    device: torch.device,
    *,
    randomize_noise: bool = True,
) -> np.ndarray:
    """Run GFPGAN on one aligned 512x512 RGB face."""
    face = (cropped / 255.0).astype(np.float32)
    tensor = torch.from_numpy(face.transpose(2, 0, 1)).float()
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = restorer(tensor, randomize_noise=randomize_noise)
    image = output.squeeze(0).float().detach().cpu().clamp_(-1, 1)
    image = (image + 1) / 2
    pixels = image.numpy().transpose(1, 2, 0)
    return (pixels * 255.0).round().astype(np.uint8)


def upscale_background(
    rgb: np.ndarray,
    request: GFPGANRequest,
    paths: list[Path],
    device: torch.device,
) -> np.ndarray:
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
    # The weights came down with the face models; loading takes a moment and is not reported.
    model = load_background_model(request.background, paths, device=device, half=request.half)
    return upscale_rgb(
        model,
        rgb,
        model_id=request.background,
        tile=request.tile,
        tile_pad=request.tile_pad,
        outscale=float(request.upscale),
        phase="restore",
    )
