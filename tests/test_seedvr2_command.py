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


def test_schema_exposes_shared_and_video_only_fields():
    schema = SeedVR2Engine().manifest()
    assert schema.media == ["video", "image"]
    properties = schema.params_schema["properties"]
    assert properties["dit_model"]["title"] == "扩散模型"
    assert "image" in properties["dit_model"]["media"]
    assert "seedvr2_ema_7b_sharp_fp16.safetensors" in properties["dit_model"]["enum"]
    assert properties["temporal_overlap"]["media"] == ["video"]
    assert properties["use_10bit"]["media"] == ["video"]
    assert properties["output_format"]["media"] == ["video"]
    assert schema.params_schema["groups"][1]["title"] == "画质"


def test_image_command_skips_video_flags(tmp_path: Path):
    source = tmp_path / "in.jpg"
    source.write_bytes(b"x")
    output = tmp_path / "out.jpg"
    job = JobSpec(
        input_path=str(source),
        output_path=str(output),
        media="image",
        devices=[0],
    )
    command = SeedVR2Engine().build_command(job, _context(tmp_path))
    assert command[command.index("--output") + 1] == str(output.resolve())
    assert command[command.index("--output_format") + 1] == "png"
    assert command[command.index("--batch_size") + 1] == "1"
    for flag in (
        "--temporal_overlap",
        "--prepend_frames",
        "--uniform_batch_size",
        "--chunk_size",
        "--skip_first_frames",
        "--load_cap",
        "--10bit",
        "--video_backend",
    ):
        assert flag not in command


def test_new_flags_are_omitted_until_set(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    context = _context(tmp_path)
    plain = SeedVR2Engine().build_command(
        JobSpec(input_path=str(source), output_path=str(tmp_path / "plain.mp4"), devices=[0, 1]),
        context,
    )
    for flag in (
        "--input_noise_scale",
        "--latent_noise_scale",
        "--max_resolution",
        "--10bit",
        "--compile_dit",
        "--blocks_to_swap",
        "--skip_first_frames",
        "--load_cap",
        "--vae_encode_tile_size",
        "--allow_vram_overflow",
    ):
        assert flag not in plain

    command = SeedVR2Engine().build_command(
        JobSpec(
            input_path=str(source),
            output_path=str(tmp_path / "out.mp4"),
            devices=[0, 1],
            params={
                "input_noise_scale": 0.2,
                "max_resolution": 1920,
                "use_10bit": True,
                "compile_dit": True,
                "compile_fullgraph": True,
                "blocks_to_swap": 4,
                "skip_first_frames": 2,
                "vae_encode_tile_size": 512,
                "allow_vram_overflow": True,
            },
        ),
        context,
    )
    assert command[command.index("--input_noise_scale") + 1] == "0.2"
    assert command[command.index("--max_resolution") + 1] == "1920"
    assert "--10bit" in command
    assert "--compile_dit" in command
    assert "--compile_fullgraph" in command
    assert command[command.index("--compile_backend") + 1] == "inductor"
    assert command[command.index("--blocks_to_swap") + 1] == "4"
    assert command[command.index("--dit_offload_device") + 1] == "cpu"
    assert command[command.index("--skip_first_frames") + 1] == "2"
    assert command[command.index("--vae_encode_tile_size") + 1] == "512"
    assert "--allow_vram_overflow" in command


def test_png_sequence_rejects_10bit(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    job = JobSpec(
        input_path=str(source),
        output_path=str(tmp_path / "frames"),
        params={"output_format": "png", "use_10bit": True},
    )
    with pytest.raises(VideoXError, match="10-bit"):
        SeedVR2Engine().build_command(job, _context(tmp_path))


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
