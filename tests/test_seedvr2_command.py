from pathlib import Path

from videox.engines.base import EngineContext
from videox.engines.seedvr2.engine import SeedVR2Engine
from videox.engines.seedvr2.presets import DEFAULT_DIT_MODEL
from videox.errors import VideoXError
from videox.hardware.gpus import GpuInfo
from videox.jobs.models import JobSpec
import pytest


def _context(tmp_path: Path) -> EngineContext:
    return EngineContext(
        project_root=tmp_path,
        python_executable="python",
        model_dir=tmp_path / "models" / "SEEDVR2",
        gpus=[
            GpuInfo(0, "RTX 4080", 16376),
            GpuInfo(1, "RTX 3080", 20480),
        ],
    )


def test_command_targets_both_gpus_and_fp8(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "out.video-only.mp4"
    job = JobSpec(
        input_path=str(source),
        output_path=str(output),
        devices=[0, 1],
        params={"resolution": 720},
    )
    command = SeedVR2Engine().build_command(job, _context(tmp_path))
    assert command[1:4] == ["-u", "-m", "videox.engines.seedvr2.progress_entry"]
    assert command[command.index("--cuda_device") + 1] == "0,1"
    assert command[command.index("--batch_size") + 1] == "9"
    assert command[command.index("--resolution") + 1] == "720"
    assert command[command.index("--dit_model") + 1] == DEFAULT_DIT_MODEL
    assert command[command.index("--temporal_overlap") + 1] == "3"
    assert command[command.index("--model_dir") + 1] == str(tmp_path / "models" / "SEEDVR2")
    assert "--vae_encode_tiled" in command
    assert "--vae_decode_tiled" in command
    assert "--uniform_batch_size" in command
    assert "--video_backend" in command
    assert command[command.index("--video_backend") + 1] == "ffmpeg"
    assert command[command.index("--attention_mode") + 1] == "sdpa"


def test_twelve_gigabyte_pair_uses_batch_five(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    context = _context(tmp_path)
    context.gpus = [GpuInfo(0, "small", 12288), GpuInfo(1, "small", 16376)]
    job = JobSpec(input_path=str(source), output_path=str(tmp_path / "out.mp4"))
    command = SeedVR2Engine().build_command(job, context)
    assert command[command.index("--cuda_device") + 1] == "0,1"
    assert command[command.index("--batch_size") + 1] == "5"


def test_batch_size_must_be_four_n_plus_one(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    job = JobSpec(
        input_path=str(source),
        output_path=str(tmp_path / "out.mp4"),
        params={"batch_size": 4},
    )
    with pytest.raises(VideoXError, match="4n\\+1"):
        SeedVR2Engine().build_command(job, _context(tmp_path))
