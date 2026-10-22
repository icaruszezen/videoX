"""Chunked RealBasicVSR inference in the main environment."""

from __future__ import annotations

import json
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
from videox.media.ffmpeg import ffmpeg_executable, ffprobe_executable


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
    reader = _FrameReader(request.input_path)
    writer: _FrameWriter | None = None
    done = 0
    total = reader.frame_count
    try:
        while True:
            batch = reader.read_many(request.max_seq_len)
            if not batch:
                break
            output = _enhance(model, batch, reader.width, reader.height)
            if writer is None:
                writer = _FrameWriter(request.output_path, output.shape[2], output.shape[1], reader.fps)
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
    _batch, _time, _channels, height, width = video.shape
    pad_h = (4 - height % 4) % 4
    pad_w = (4 - width % 4) % 4
    if height + pad_h < 64:
        pad_h += 64 - (height + pad_h)
    if width + pad_w < 64:
        pad_w += 64 - (width + pad_w)
    if pad_h or pad_w:
        video = F.pad(video, (0, pad_w, 0, pad_h))
    return video, height, width


class _FrameReader:
    def __init__(self, path: Path) -> None:
        self.width, self.height, self.fps, self.frame_count = _probe(path)
        self._frame_bytes = self.width * self.height * 3
        self._process = subprocess.Popen(
            [
                ffmpeg_executable(),
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read_many(self, count: int) -> list[bytes]:
        frames: list[bytes] = []
        assert self._process.stdout is not None
        for _ in range(count):
            blob = self._process.stdout.read(self._frame_bytes)
            if len(blob) != self._frame_bytes:
                break
            frames.append(blob)
        return frames

    def close(self) -> None:
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._process.wait()


class _FrameWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
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


def _probe(path: Path) -> tuple[int, int, float, int]:
    completed = subprocess.run(
        [
            ffprobe_executable(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise VideoXError(f"无法读取视频信息: {path}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoXError(f"ffprobe 输出不是 JSON: {path}") from exc
    video = next(
        (stream for stream in payload.get("streams") or [] if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise VideoXError(f"视频里没有画面: {path}")
    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoXError(f"读不到视频分辨率: {path}") from exc
    fps = _parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1")
    frames = _frame_count(video, payload.get("format") or {}, fps)
    return width, height, fps, frames


def _parse_rate(value: object) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            den = float(denominator)
            return float(numerator) / den if den else 25.0
        except ValueError:
            return 25.0
    try:
        return float(text)
    except ValueError:
        return 25.0


def _frame_count(stream: dict, fmt: dict, fps: float) -> int:
    raw = stream.get("nb_frames")
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = 0
    if count > 0:
        return count
    try:
        duration = float(fmt.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    if duration > 0 and fps > 0:
        return max(int(round(duration * fps)), 1)
    return 1
