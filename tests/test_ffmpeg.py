from pathlib import Path

from videox.media.ffmpeg import remux_audio_args


def test_remux_copies_original_audio_by_default():
    command = remux_audio_args(
        Path("video-only.mp4"),
        Path("original.mp4"),
        Path("out.mp4"),
        ffmpeg="ffmpeg",
    )
    assert command[:3] == ["ffmpeg", "-y", "-i"]
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in command
    assert command[command.index("-c:a") + 1] == "copy"
    assert command[-2] == "-shortest"
    assert command[-1] == "out.mp4"


def test_remux_can_reencode_audio_when_copy_fails():
    command = remux_audio_args(
        Path("video-only.mp4"),
        Path("original.mp4"),
        Path("out.mp4"),
        reencode_audio=True,
    )
    assert command[command.index("-c:a") + 1] == "aac"
