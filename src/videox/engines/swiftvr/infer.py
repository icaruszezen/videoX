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
    short_side: int = 0


def run_swiftvr(request: SwiftRequest) -> None:
    if request.clip_len % 4 != 0:
        raise VideoXError(f"分块长度必须是 4 的倍数，当前是 {request.clip_len}")
    if not request.input_path.is_file():
        raise VideoXError(f"输入文件不存在: {request.input_path}")
    emit_progress("download", 0, 1)
    ensure_weights(request.checkpoint)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    import torch
    from swiftvr import SwiftVRPipeline

    dtype = request.dtype
    # is_bf16_supported() also counts emulation, which pre-Ampere cards run very slowly.
    if dtype == "bfloat16" and torch.cuda.get_device_capability()[0] < 8:
        print("这张显卡不支持原生 bfloat16，改用 float16。", flush=True)
        dtype = "float16"
    pipe = SwiftVRPipeline.from_pretrained(str(request.checkpoint)).to(
        "cuda",
        dtype=dtype,
        attention_backend="auto",
        torch_compile=False,
    )
    emit_progress("load", 1, 1)
    from swiftvr.io import get_video_info

    raw_total, lq_h, lq_w, _fps = get_video_info(str(request.input_path))
    kept = 4 * ((raw_total - 1) // 4) + 1
    resolution = _output_size(lq_w, lq_h, request.short_side)
    out_w, out_h = resolution or (lq_w * request.upscale, lq_h * request.upscale)
    print(f"SwiftVR 将处理 {kept}/{raw_total} 帧，画面 {lq_w}x{lq_h}，输出 {out_w}x{out_h}", flush=True)
    _report_chunks(pipe, kept, request.clip_len)
    _write_at_source_rate(request.input_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    pipe.restore_video(
        str(request.input_path),
        str(request.output_path),
        resolution=resolution,
        upscale=request.upscale,
        clip_len=request.clip_len,
        dit_overlap=0,
        # Upstream's CLI default (x265 CRF about 20); 90 wrote near-lossless, very large files.
        quality=60,
        png_save=False,
        queue_size=3,
        verbose=True,
    )
    if not request.output_path.is_file():
        raise VideoXError(f"引擎没有写出视频: {request.output_path}")
    _publish_ends(request.input_path, request.output_path)
    emit_progress("restore", 1, 1)


def _output_size(lq_w: int, lq_h: int, short_side: int) -> tuple[int, int] | None:
    """(width, height) with the given short side and the source aspect, rounded to even."""
    if short_side <= 0:
        return None
    if lq_w >= lq_h:
        return max(2, round(lq_w * short_side / lq_h / 2) * 2), short_side
    return short_side, max(2, round(lq_h * short_side / lq_w / 2) * 2)


def _report_chunks(pipe, frames: int, clip_len: int) -> None:
    """Upstream prints no progress; count the chunks it decodes, one call per chunk."""
    from swiftvr.streaming.chunk import build_chunk_specs

    total = max(1, len(build_chunk_specs(frames, clip_len)))
    decode = pipe.tae_stream.decode_chunk_fixed
    done = [0]

    def counted(z, spec):
        out = decode(z, spec)
        done[0] += 1
        emit_progress("restore", min(done[0], total), total)
        return out

    pipe.tae_stream.decode_chunk_fixed = counted
    emit_progress("restore", 0, total)


def _write_at_source_rate(source: Path) -> None:
    """Stamp the output with the source's exact rate (24000/1001, not imageio's 23.98)."""
    from videox.media.ffmpeg import probe_media

    try:
        rate = probe_media(source).frame_rate
    except VideoXError:
        rate = None
    if not rate:
        return
    import imageio
    import swiftvr.runner as runner
    from swiftvr.io import quality_to_crf

    def open_writer(output_path, fps=8, video_format="", preset="", quality=85):
        # Same as upstream's open_stream_video_writer, plus an input -r; ffmpeg keeps the last one.
        extra = ["-preset", str(preset)] if preset else []
        pixels = "yuv444p" if video_format == "yuv444p" else "yuv420p"
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx265",
            pixelformat=pixels,
            macro_block_size=None,
            ffmpeg_params=["-crf", str(quality_to_crf(quality))] + extra,
            input_params=["-r", rate],
        )

    runner.open_stream_video_writer = open_writer


def _publish_ends(source: Path, output: Path) -> None:
    directory = os.environ.get("VIDEOX_PREVIEW_DIR")
    ffmpeg = tool_executable("ffmpeg")
    if not directory or ffmpeg is None:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _grab(ffmpeg, source, root / "source.jpg")
    if _grab(ffmpeg, output, root / "output.jpg"):
        # Both pictures are the first frame.
        emit_preview(1)


def _grab(ffmpeg: str, video: Path, jpeg: Path) -> bool:
    completed = subprocess.run(
        [ffmpeg, "-y", "-i", str(video), "-frames:v", "1", str(jpeg)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0 and jpeg.is_file()
