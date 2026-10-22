"""Face detection, alignment and paste-back around GFPGAN.

Adapted from facexlib (MIT, Copyright (c) 2020 Xintao Wang), tag v0.3.0:
https://github.com/xinntao/facexlib/blob/v0.3.0/facexlib/utils/face_restoration_helper.py
(``get_face_landmarks_5``, ``align_warp_face``, ``get_inverse_affine`` and the
square-mask branch of ``paste_faces_to_input_image``; the face parsing model,
blur padding and 3-point template are not used). facexlib works on BGR images;
alignment and paste-back are per-channel, so they run on RGB here and only the
detector sees BGR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from videox.engines.gfpgan.arch import RetinaFace
from videox.engines.gfpgan.presets import DETECT_CONFIDENCE, FACE_SIZE, MAX_DETECT_EDGE
from videox.errors import VideoXError

# Standard five landmarks of FFHQ faces at 512x512: eyes, nose, mouth corners.
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ]
) * (FACE_SIZE / 512.0)
# facexlib fills outside the image with gray (135, 133, 132) in BGR.
BORDER_RGB = (132, 133, 135)


def cv2_module():
    try:
        import cv2
    except ImportError as exc:
        raise VideoXError("主环境缺少 opencv-python，请运行 videox setup") from exc
    return cv2


@dataclass(frozen=True)
class DetectedFace:
    box: np.ndarray  # x1, y1, x2, y2, score in input pixels
    landmarks: np.ndarray  # 5x2 float32 in input pixels


def detect_faces(
    detector: RetinaFace,
    bgr: np.ndarray,
    *,
    only_center_face: bool,
    eye_dist_threshold: float | None,
    max_edge: int = MAX_DETECT_EDGE,
) -> list[DetectedFace]:
    """``get_face_landmarks_5`` with ``resize=None``: detect, drop tiny faces, optionally keep the centre one.

    facexlib detects at the full input size. Inputs whose long edge is above
    ``max_edge`` are detected on a copy shrunk to ``max_edge`` and the results
    scaled back, which keeps the detector's memory bounded.
    """
    height, width = bgr.shape[:2]
    image = bgr
    if max(height, width) > max_edge:
        cv2 = cv2_module()
        ratio = max_edge / max(height, width)
        small_w, small_h = max(round(width * ratio), 1), max(round(height * ratio), 1)
        image = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    rows = detector.detect_faces(image, DETECT_CONFIDENCE)
    if image is not bgr:
        scale_x = np.float32(width / image.shape[1])
        scale_y = np.float32(height / image.shape[0])
        rows = rows.copy()
        rows[:, 0:4:2] *= scale_x
        rows[:, 1:4:2] *= scale_y
        rows[:, 5:15:2] *= scale_x
        rows[:, 6:15:2] *= scale_y
    faces: list[DetectedFace] = []
    for row in rows:
        # Side faces and tiny faces have close eyes; upstream skips them.
        eye_dist = np.linalg.norm([row[5] - row[7], row[6] - row[8]])
        if eye_dist_threshold is not None and eye_dist < eye_dist_threshold:
            continue
        landmarks = np.array([[row[i], row[i + 1]] for i in range(5, 15, 2)])
        faces.append(DetectedFace(box=row[0:5], landmarks=landmarks))
    if faces and only_center_face:
        faces = [faces[center_face_index(faces, height, width)]]
    return faces


def center_face_index(faces: list[DetectedFace], height: int, width: int) -> int:
    center = np.array([width / 2, height / 2])
    distances = []
    for face in faces:
        box = face.box
        face_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        distances.append(np.linalg.norm(face_center - center))
    return distances.index(min(distances))


def align_face(rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp one face onto the FFHQ template. Returns the 2x3 affine matrix and the 512x512 crop."""
    cv2 = cv2_module()
    # LMEDS matches skimage's similarity transform, as upstream notes.
    affine_matrix = cv2.estimateAffinePartial2D(landmarks, FACE_TEMPLATE, method=cv2.LMEDS)[0]
    if affine_matrix is None:
        raise VideoXError("人脸对齐失败")
    cropped = cv2.warpAffine(
        rgb,
        affine_matrix,
        (FACE_SIZE, FACE_SIZE),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_RGB,
    )
    return affine_matrix, cropped


def inverse_affine(affine_matrix: np.ndarray, upscale: int) -> np.ndarray:
    cv2 = cv2_module()
    return cv2.invertAffineTransform(affine_matrix) * upscale


def paste_faces(
    background: np.ndarray,
    restored_faces: list[np.ndarray],
    inverse_affines: list[np.ndarray],
    *,
    input_size: tuple[int, int],
    upscale: int,
) -> np.ndarray:
    """Blend restored faces into the upscaled background with a soft square mask.

    ``input_size`` is (height, width) of the original image; ``background`` is
    resized to ``int(size * upscale)`` first, as upstream does.
    """
    cv2 = cv2_module()
    height, width = input_size
    h_up, w_up = int(height * upscale), int(width * upscale)
    image = cv2.resize(background, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
    for restored_face, matrix in zip(restored_faces, inverse_affines, strict=True):
        matrix = matrix.copy()
        # Shift by half an output pixel for more precise back alignment.
        if upscale > 1:
            matrix[:, 2] += 0.5 * upscale
        inv_restored = cv2.warpAffine(restored_face, matrix, (w_up, h_up))
        mask = np.ones((FACE_SIZE, FACE_SIZE), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, matrix, (w_up, h_up))
        # Remove the black border left by the warp.
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((int(2 * upscale), int(2 * upscale)), np.uint8))
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        # The blend edge grows with the face.
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)[:, :, None]
        image = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * image
    # Truncates like upstream's astype; the blend stays within 0..255.
    return np.clip(image, 0, 255).astype(np.uint8)
