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
    fps: float | None = None
    # Exact frame rate as ffprobe reports it, e.g. "30000/1001".
    frame_rate: str | None = None
    video_codec: str | None = None


# MP4/MOV players (QuickTime, Safari, iOS) only open HEVC tagged as hvc1.
_HVC1_SUFFIXES = {".mp4", ".m4v", ".mov"}


def _muxer_args(dest: Path) -> list[str]:
    # ffmpeg picks its ipod muxer for .m4v, which refuses HEVC; .m4v is plain MP4.
    return ["-f", "mp4"] if dest.suffix.lower() == ".m4v" else []


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
    fps = frame_rate = codec = None
    if video is not None:
        try:
            width = int(video["width"])
            height = int(video["height"])
        except (KeyError, TypeError, ValueError):
            width = height = None
        for key in ("avg_frame_rate", "r_frame_rate"):
            parsed = parse_rate(video.get(key))
            if parsed is not None:
                fps, frame_rate = parsed
                break
        codec = video.get("codec_name") or None
    return MediaInfo(
        has_audio=has_audio,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        frame_rate=frame_rate,
        video_codec=codec,
    )


def parse_rate(value: object) -> tuple[float, str] | None:
    """Parse an ffprobe rate such as "30000/1001"; 0/0 and garbage give None."""
    text = str(value or "").strip()
    numerator, _, denominator = text.partition("/")
    try:
        top = float(numerator)
        bottom = float(denominator) if denominator else 1.0
    except ValueError:
        return None
    if top <= 0 or bottom <= 0:
        return None
    return top / bottom, text


def remux_audio_args(
    video_only: Path,
    original: Path,
    dest: Path,
    *,
    ffmpeg: str = "ffmpeg",
    reencode_audio: bool = False,
    audio_offset: float = 0.0,
    max_duration: float | None = None,
    hevc: bool = False,
) -> list[str]:
    command = [ffmpeg, "-y", "-i", str(video_only)]
    # Both options apply to the original only: -ss skips the audio the engine
    # skipped in the picture, -t stops the audio where the restored video ends.
    if audio_offset > 0:
        command.extend(["-ss", f"{audio_offset:.6f}"])
    if max_duration is not None and max_duration > 0:
        command.extend(["-t", f"{max_duration:.6f}"])
    command.extend(["-i", str(original), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"])
    if hevc and dest.suffix.lower() in _HVC1_SUFFIXES:
        command.extend(["-tag:v", "hvc1"])
    if reencode_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        command.extend(["-c:a", "copy"])
    if max_duration is None:
        # Without a known video length, fall back to trimming at the shorter stream.
        command.append("-shortest")
    command.extend([*_muxer_args(dest), str(dest)])
    return command


def remux_audio(
    video_only: Path,
    original: Path,
    dest: Path,
    *,
    audio_offset: float = 0.0,
    max_duration: float | None = None,
    hevc: bool = False,
) -> None:
    ffmpeg = ffmpeg_executable()
    dest.parent.mkdir(parents=True, exist_ok=True)
    options = {"audio_offset": audio_offset, "max_duration": max_duration, "hevc": hevc}
    copied = subprocess.run(
        remux_audio_args(video_only, original, dest, ffmpeg=ffmpeg, **options),
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if copied.returncode == 0:
        return
    encoded = subprocess.run(
        remux_audio_args(video_only, original, dest, ffmpeg=ffmpeg, reencode_audio=True, **options),
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if encoded.returncode != 0:
        detail = (encoded.stderr or copied.stderr or "").strip()
        raise VideoXError(f"回封装音轨失败: {dest}\n{detail}")


def transmux_video(video_only: Path, dest: Path, *, hevc: bool = False) -> None:
    """Copy the silent video into the destination container."""
    retag = hevc and dest.suffix.lower() in _HVC1_SUFFIXES
    if dest.suffix.lower() == ".mp4" and not retag:
        dest.parent.mkdir(parents=True, exist_ok=True)
        video_only.replace(dest)
        return
    ffmpeg = ffmpeg_executable()
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-i", str(video_only), "-c", "copy"]
    if retag:
        command.extend(["-tag:v", "hvc1"])
    completed = subprocess.run(
        [*command, *_muxer_args(dest), str(dest)],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise VideoXError(f"封装成片失败: {dest}\n{detail}")
    video_only.unlink(missing_ok=True)
