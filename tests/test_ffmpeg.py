from pathlib import Path

from videox.media.ffmpeg import parse_rate, remux_audio_args


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


def test_remux_bounds_audio_by_the_video_instead_of_cutting_the_video():
    command = remux_audio_args(
        Path("video-only.mp4"),
        Path("original.mp4"),
        Path("out.mp4"),
        max_duration=6.0,
    )
    original = command.index("original.mp4")
    assert command[original - 1] == "-i"
    assert command[command.index("-t") + 1] == "6.000000"
    assert command.index("-t") < original
    assert "-shortest" not in command


def test_remux_skips_the_audio_the_engine_skipped():
    command = remux_audio_args(
        Path("video-only.mp4"),
        Path("original.mp4"),
        Path("out.mp4"),
        audio_offset=12.0,
        max_duration=4.0,
    )
    original = command.index("original.mp4")
    assert command[command.index("-ss") + 1] == "12.000000"
    assert command.index("-ss") < original
    assert command.index("-ss") > command.index("video-only.mp4")


def test_hevc_is_tagged_for_mp4_players():
    tagged = remux_audio_args(Path("v.mp4"), Path("o.mp4"), Path("out.mp4"), hevc=True)
    assert tagged[tagged.index("-tag:v") + 1] == "hvc1"
    untagged = remux_audio_args(Path("v.mp4"), Path("o.mp4"), Path("out.mkv"), hevc=True)
    assert "-tag:v" not in untagged


def test_parse_rate_keeps_ntsc_rates_exact():
    assert parse_rate("30000/1001") == (30000 / 1001, "30000/1001")
    assert parse_rate("25") == (25.0, "25")
    assert parse_rate("0/0") is None
    assert parse_rate(None) is None


def test_remux_can_reencode_audio_when_copy_fails():
    command = remux_audio_args(
        Path("video-only.mp4"),
        Path("original.mp4"),
        Path("out.mp4"),
        reencode_audio=True,
    )
    assert command[command.index("-c:a") + 1] == "aac"
