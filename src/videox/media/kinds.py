from __future__ import annotations

from pathlib import Path

from videox.errors import VideoXError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def resolve_media(requested: str, path: Path) -> str:
    if requested not in {"video", "image"}:
        raise VideoXError(f"未知媒体类型: {requested}")
    image = is_image_path(path)
    if requested == "image" and not image:
        raise VideoXError(f"图片任务需要图片文件: {path.name}")
    if requested == "video" and image:
        raise VideoXError(f"这是图片文件，请用图片任务处理: {path.name}")
    return requested
