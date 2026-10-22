"""Decode a video into raw RGB frames with ffmpeg, the same way for every engine.

ffmpeg reads any container it knows, turns portrait phone video upright and
resamples variable frame rate clips to the constant rate the output is written at.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from videox.errors import VideoXError
from videox.media.ffmpeg import ffmpeg_executable, ffprobe_executable


# Restored frames are usually HD-sized, which players decode as BT.709; convert
# RGB with that matrix and tag it, instead of swscale's untagged BT.601.
BT709_OUTPUT_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
)


class FrameReader:
    def __init__(self, path: Path) -> None:
        probe = probe_video(path)
        self.width, self.height = probe.width, probe.height
        self.fps, self.rate, self.sar = probe.fps, probe.rate, probe.sar
        self.frame_count = probe.frame_count
        self._frame_bytes = self.width * self.height * 3
        # Untagged HD: players assume BT.709, while ffmpeg would convert with BT.601.
        matrix = ["-vf", "scale=in_color_matrix=bt709,format=rgb24"] if probe.untagged_hd else []
        # A file, not a pipe: a long run of decode warnings cannot block ffmpeg.
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [
                ffmpeg_executable(),
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                # Decode at the same constant rate the writer stamps, so variable
                # frame rate phone clips do not come out longer than the source.
                "-r",
                self.rate,
                *matrix,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=self._stderr,
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

    def finish(self) -> None:
        """After the last frame: report a decode failure instead of a short video."""
        code = self._process.wait()
        if code == 0:
            return
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(lines[-20:])
        raise VideoXError(f"读取视频失败 (ffmpeg {code}): {tail}")

    def close(self) -> None:
        # Never raises: closing early makes ffmpeg exit non-zero on purpose.
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._process.wait()
        self._stderr.close()


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    rate: str
    sar: str | None
    frame_count: int
    untagged_hd: bool = False


def probe_video(path: Path) -> VideoProbe:
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
    # mpv, VLC and browsers pick the matrix of untagged video by its coded size.
    untagged_hd = str(video.get("color_space") or "unknown") in {"unknown", "reserved"} and (
        width >= 1280 or height > 576
    )
    sar = _sample_aspect(video)
    if _quarter_turn(video):
        # ffmpeg rotates portrait phone video while decoding, so the frames
        # arrive with width and height swapped, and the pixel aspect inverted.
        width, height = height, width
        if sar:
            numerator, _, denominator = sar.partition("/")
            sar = f"{denominator}/{numerator}"
    rate = _choose_rate(video)
    fps = _parse_rate(rate)
    frames = _frame_count(video, payload.get("format") or {}, fps)
    return VideoProbe(width, height, fps, rate, sar, frames, untagged_hd)


def _quarter_turn(stream: dict) -> bool:
    values = [item.get("rotation") for item in stream.get("side_data_list") or [] if isinstance(item, dict)]
    values.append((stream.get("tags") or {}).get("rotate"))  # ffprobe before 5.0
    for value in values:
        try:
            return round(abs(float(value))) % 180 == 90
        except (TypeError, ValueError):
            continue
    return False


def _choose_rate(stream: dict) -> str:
    for key in ("avg_frame_rate", "r_frame_rate"):
        text = str(stream.get(key) or "")
        if _parse_rate(text, 0.0) > 0:
            return text
    return "25/1"


def _sample_aspect(stream: dict) -> str | None:
    """Non-square pixels (DVD and broadcast rips) as "num/den", else None."""
    text = str(stream.get("sample_aspect_ratio") or "")
    numerator, _, denominator = text.partition(":")
    try:
        num, den = int(numerator), int(denominator)
    except ValueError:
        return None
    if num <= 0 or den <= 0 or num == den:
        return None
    return f"{num}/{den}"


def _parse_rate(value: object, fallback: float = 25.0) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            den = float(denominator)
            rate = float(numerator) / den if den else fallback
        except ValueError:
            return fallback
        return rate if rate > 0 else fallback
    try:
        rate = float(text)
    except ValueError:
        return fallback
    return rate if rate > 0 else fallback


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
