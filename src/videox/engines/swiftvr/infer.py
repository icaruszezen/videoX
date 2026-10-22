"""SwiftVR inference through the upstream pipeline installed in its own venv."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from videox.engines.progress import emit_preview, emit_progress
from videox.engines.swiftvr.weights import ensure_weights
from videox.errors import VideoXError
from videox.paths import tool_executable


@dataclass(frozen=True)
class SwiftRequest:
    input_path: Path
    output_path: Path
    checkpoint: Path
    upscale: int
    clip_len: int
    dtype: str


def run_swiftvr(request: SwiftRequest) -> None:
    if request.clip_len % 4 != 0:
        raise VideoXError(f"分块长度必须是 4 的倍数，当前是 {request.clip_len}")
    if not request.input_path.is_file():
        raise VideoXError(f"输入文件不存在: {request.input_path}")
    _patch_tqdm()
    emit_progress("download", 0, 1)
    ensure_weights(request.checkpoint)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    import torch
    from swiftvr import SwiftVRPipeline

    dtype = request.dtype
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("这张显卡不支持 bfloat16，改用 float16。", flush=True)
        dtype = "float16"
    pipe = SwiftVRPipeline.from_pretrained(str(request.checkpoint)).to(
        "cuda",
        dtype=dtype,
        attention_backend="auto",
        torch_compile=False,
    )
    emit_progress("load", 1, 1)
    emit_progress("restore", 0, 1)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = pipe.restore_video(
        str(request.input_path),
        str(request.output_path),
        upscale=request.upscale,
        clip_len=request.clip_len,
        dit_overlap=0,
        quality=90,
        png_save=False,
        queue_size=3,
        verbose=True,
    )
    if not request.output_path.is_file():
        raise VideoXError(f"引擎没有写出视频: {request.output_path}")
    frames = int(stats.get("frames") or 1) if isinstance(stats, dict) else 1
    _publish_ends(request.input_path, request.output_path, max(frames, 1))
    emit_progress("restore", 1, 1)


def _patch_tqdm() -> None:
    import tqdm as tqdm_module

    original = tqdm_module.tqdm

    class ReportingTqdm(original):
        def update(self, n=1):
            result = super().update(n)
            total = self.total or 0
            if total:
                emit_progress("restore", float(self.n or 0), float(total))
            return result

    tqdm_module.tqdm = ReportingTqdm


def _publish_ends(source: Path, output: Path, frame: int) -> None:
    directory = os.environ.get("VIDEOX_PREVIEW_DIR")
    ffmpeg = tool_executable("ffmpeg")
    if not directory or ffmpeg is None:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _grab(ffmpeg, source, root / "source.jpg")
    if _grab(ffmpeg, output, root / "output.jpg"):
        emit_preview(frame)


def _grab(ffmpeg: str, video: Path, jpeg: Path) -> bool:
    completed = subprocess.run(
        [ffmpeg, "-y", "-i", str(video), "-frames:v", "1", str(jpeg)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0 and jpeg.is_file()
