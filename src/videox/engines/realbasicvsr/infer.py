"""Chunked RealBasicVSR inference in the main environment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from videox.engines.preview_io import publish_pair
from videox.engines.progress import emit_progress
from videox.engines.realbasicvsr.arch import load_realbasicvsr
from videox.engines.realbasicvsr.weights import ensure_checkpoint
from videox.errors import VideoXError
from videox.media.ffmpeg import ffmpeg_executable
from videox.media.frames import BT709_OUTPUT_FILTER, FrameReader


@dataclass(frozen=True)
class RealBasicRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    max_seq_len: int


def run_realbasicvsr(request: RealBasicRequest) -> None:
    if request.max_seq_len < 1:
        raise VideoXError("一次处理帧数需要是正整数")
    if not request.input_path.is_file():
        raise VideoXError(f"输入文件不存在: {request.input_path}")
    if not torch.cuda.is_available():
        raise VideoXError("RealBasicVSR 需要 CUDA 显卡")
    emit_progress("download", 0, 1)
    checkpoint = ensure_checkpoint(request.model_dir)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    model = load_realbasicvsr(checkpoint, torch.device("cuda"))
    emit_progress("load", 1, 1)
    reader = FrameReader(request.input_path)
    writer: _FrameWriter | None = None
    done = 0
    total = reader.frame_count
    try:
        while True:
            batch = reader.read_many(request.max_seq_len)
            if not batch:
                break
            try:
                output = _enhance(model, batch, reader.width, reader.height)
            except torch.cuda.OutOfMemoryError as exc:
                raise VideoXError(
                    f"显存不足：{reader.width}x{reader.height} 的原片会输出 {reader.width * 4}x{reader.height * 4}。"
                    "显存主要随原片分辨率增长，1080p 原片在 16GB 及以下的显卡上即使一次 1 帧也放不下；"
                    "请先把原片缩到 720p 或更低，或调小一次处理帧数。"
                ) from exc
            if writer is None:
                writer = _FrameWriter(
                    request.output_path, output.shape[2], output.shape[1], reader.rate, sar=reader.sar
                )
            for frame in output:
                writer.write(np.ascontiguousarray(frame).tobytes())
            done += len(batch)
            if total < done:
                total = done
            emit_progress("restore", done, total)
            last = np.ascontiguousarray(output[-1])
            source = np.frombuffer(batch[-1], dtype=np.uint8).reshape(reader.height, reader.width, 3)
            publish_pair(
                os.environ.get("VIDEOX_PREVIEW_DIR"),
                done,
                last.tobytes(),
                int(last.shape[1]),
                int(last.shape[0]),
                source.tobytes(),
                reader.width,
                reader.height,
            )
        reader.finish()
    finally:
        reader.close()
        if writer is not None:
            writer.close()
    if writer is None or not request.output_path.is_file():
        raise VideoXError(f"引擎没有写出视频: {request.output_path}")
    emit_progress("restore", 1, 1)


def _enhance(model, frames: list[bytes], width: int, height: int):
    array = np.stack([np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3) for frame in frames])
    tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).unsqueeze(0).float().div_(255)
    tensor, origin_h, origin_w = _pad(tensor)
    with torch.inference_mode():
        output = model(tensor.to(device="cuda", non_blocking=True))
    output = output[..., : origin_h * 4, : origin_w * 4]
    output = output.squeeze(0).clamp_(0, 1).mul_(255).round_().byte()
    return output.permute(0, 2, 3, 1).cpu().numpy()


def _pad(video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Only tiny frames need padding (SPyNet wants 64 px); repeat the edge, not black."""
    batch, time, channels, height, width = video.shape
    pad_h = max(0, 64 - height)
    pad_w = max(0, 64 - width)
    if pad_h or pad_w:
        # Replicate padding needs a 4D tensor.
        flat = video.reshape(batch * time, channels, height, width)
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
        video = flat.reshape(batch, time, channels, height + pad_h, width + pad_w)
    return video, height, width


class _FrameWriter:
    def __init__(self, path: Path, width: int, height: int, rate: str, *, sar: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        filters = BT709_OUTPUT_FILTER
        if sar:
            filters += f",setsar={sar}"
        self._process = subprocess.Popen(
            [
                ffmpeg_executable(),
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                rate,
                "-i",
                "-",
                "-an",
                "-vf",
                filters,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "16",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, blob: bytes) -> None:
        if self._process.stdin is None:
            raise VideoXError("写出视频失败")
        self._process.stdin.write(blob)

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        error = self._process.stderr.read().decode("utf-8", "replace") if self._process.stderr else ""
        code = self._process.wait()
        if code != 0:
            raise VideoXError(f"写出视频失败: {error.strip()}")
