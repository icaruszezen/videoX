from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from videox.errors import VideoXError
from videox.paths import tool_executable


@dataclass(frozen=True)
class MediaInfo:
    has_audio: bool
    width: int | None = None
    height: int | None = None
    duration: float | None = None


def ffmpeg_executable() -> str:
    found = tool_executable("ffmpeg")
    if not found:
        raise VideoXError("找不到 ffmpeg。请先运行 videox setup。")
    return found


def ffprobe_executable() -> str:
    found = tool_executable("ffprobe")
    if not found:
        raise VideoXError("找不到 ffprobe。请先运行 videox setup。")
    return found


def probe_media(path: Path) -> MediaInfo:
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
        detail = (completed.stderr or completed.stdout).strip()
        raise VideoXError(f"无法读取视频信息: {path}\n{detail}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoXError(f"ffprobe 输出不是 JSON: {path}") from exc
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    duration = None
    raw_duration = (payload.get("format") or {}).get("duration")
    if raw_duration is not None:
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = None
    width = height = None
    if video is not None:
        try:
            width = int(video["width"])
            height = int(video["height"])
        except (KeyError, TypeError, ValueError):
            width = height = None
    return MediaInfo(has_audio=has_audio, width=width, height=height, duration=duration)


def remux_audio_args(
    video_only: Path,
    original: Path,
    dest: Path,
    *,
    ffmpeg: str = "ffmpeg",
    reencode_audio: bool = False,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if reencode_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        command.extend(["-c:a", "copy"])
    command.extend(["-shortest", str(dest)])
    return command


def remux_audio(video_only: Path, original: Path, dest: Path) -> None:
    ffmpeg = ffmpeg_executable()
    dest.parent.mkdir(parents=True, exist_ok=True)
    copied = subprocess.run(
        remux_audio_args(video_only, original, dest, ffmpeg=ffmpeg),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if copied.returncode == 0:
        return
    encoded = subprocess.run(
        remux_audio_args(video_only, original, dest, ffmpeg=ffmpeg, reencode_audio=True),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if encoded.returncode != 0:
        detail = (encoded.stderr or copied.stderr or "").strip()
        raise VideoXError(f"回封装音轨失败: {dest}\n{detail}")


def transmux_video(video_only: Path, dest: Path) -> None:
    """Copy the silent video into a non-mp4 container."""
    if dest.suffix.lower() == ".mp4":
        dest.parent.mkdir(parents=True, exist_ok=True)
        video_only.replace(dest)
        return
    ffmpeg = ffmpeg_executable()
    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [ffmpeg, "-y", "-i", str(video_only), "-c", "copy", str(dest)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise VideoXError(f"封装成片失败: {dest}\n{detail}")
    video_only.unlink(missing_ok=True)
