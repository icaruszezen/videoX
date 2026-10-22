import shutil
import subprocess
from pathlib import Path

import pytest

from videox.media.frames import FrameReader, probe_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


def _clip(path: Path, *extra: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30000/1001",
         "-frames:v", "10", *extra, str(path)],
        check=True,
    )
    return path


def _read_all(path: Path) -> tuple[FrameReader, list[bytes]]:
    reader = FrameReader(path)
    frames: list[bytes] = []
    try:
        while batch := reader.read_many(4):
            frames.extend(batch)
        reader.finish()
    finally:
        reader.close()
    return reader, frames


@pytest.mark.parametrize("suffix", [".ts", ".m4v", ".mkv"])
def test_reader_opens_common_containers_at_the_exact_rate(tmp_path: Path, suffix: str):
    reader, frames = _read_all(_clip(tmp_path / f"clip{suffix}", "-pix_fmt", "yuv420p"))
    assert (reader.width, reader.height, reader.rate) == (64, 48, "30000/1001")
    assert len(frames) == 10
    assert all(len(frame) == 64 * 48 * 3 for frame in frames)


def test_reader_turns_portrait_video_upright(tmp_path: Path):
    plain = _clip(tmp_path / "plain.mp4", "-pix_fmt", "yuv420p")
    rotated = tmp_path / "rotated.mp4"
    tagged = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(plain), "-c", "copy", str(rotated)],
        check=False,
    )
    if tagged.returncode != 0:
        pytest.skip("ffmpeg before 6.0 has no -display_rotation")
    reader, frames = _read_all(rotated)
    assert (reader.width, reader.height) == (48, 64)
    assert len(frames) == 10 and len(frames[0]) == 48 * 64 * 3


def test_rotated_video_gets_the_inverse_pixel_aspect(tmp_path: Path):
    plain = _clip(tmp_path / "plain.mp4", "-vf", "setsar=8/9", "-pix_fmt", "yuv420p")
    rotated = tmp_path / "rotated.mp4"
    tagged = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(plain), "-c", "copy", str(rotated)],
        check=False,
    )
    if tagged.returncode != 0:
        pytest.skip("ffmpeg before 6.0 has no -display_rotation")
    assert probe_video(plain).sar == "8/9"
    probe = probe_video(rotated)
    assert (probe.width, probe.height, probe.sar) == (48, 64, "9/8")


def test_untagged_hd_is_read_with_the_bt709_matrix(tmp_path: Path):
    hd = tmp_path / "hd.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25",
         "-frames:v", "2", "-pix_fmt", "yuv420p", str(hd)],
        check=True,
    )
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(hd), "-c", "copy", "-bsf:v",
         "h264_metadata=matrix_coefficients=1", str(tagged)],
        check=True,
    )
    sd = _clip(tmp_path / "sd.mp4", "-pix_fmt", "yuv420p")
    assert probe_video(hd).untagged_hd
    assert not probe_video(tagged).untagged_hd
    assert not probe_video(sd).untagged_hd
