from __future__ import annotations

import subprocess
from pathlib import Path

from videox.engines.progress import emit_preview
from videox.paths import tool_executable


def write_jpeg(path: Path, rgb: bytes, width: int, height: int) -> bool:
    ffmpeg = tool_executable("ffmpeg")
    if ffmpeg is None or width < 1 or height < 1 or len(rgb) < width * height * 3:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-frames:v",
            "1",
            str(path),
        ],
        input=rgb,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def publish_pair(
    directory: str | None,
    frame: int,
    output_rgb: bytes,
    output_width: int,
    output_height: int,
    source_rgb: bytes | None = None,
    source_width: int = 0,
    source_height: int = 0,
) -> None:
    if not directory:
        return
    root = Path(directory)
    if not write_jpeg(root / "output.jpg", output_rgb, output_width, output_height):
        return
    if source_rgb is not None:
        write_jpeg(root / "source.jpg", source_rgb, source_width, source_height)
    emit_preview(frame)
