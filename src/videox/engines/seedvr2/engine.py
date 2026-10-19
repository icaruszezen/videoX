from __future__ import annotations

from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.seedvr2.logs import parse_seedvr2_line
from videox.engines.seedvr2.presets import (
    DEFAULT_ATTENTION_MODE,
    DEFAULT_COLOR_CORRECTION,
    DEFAULT_DIT_MODEL,
    DEFAULT_PREPEND_FRAMES,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    DEFAULT_TEMPORAL_OVERLAP,
    params_schema,
)
from videox.errors import VideoXError
from videox.hardware.gpus import batch_size_for_devices, default_devices, query_gpus
from videox.jobs.models import EngineManifest, JobSpec
from videox.paths import seedvr2_dir


class SeedVR2Engine:
    id = "seedvr2"

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="SeedVR2",
            description="一步扩散视频修复。双卡按帧分段，每张卡各加载一份 3B FP8 模型。",
            capabilities=["upscale"],
            params_schema=params_schema(),
        )

    def working_directory(self, ctx: EngineContext) -> Path | None:
        return seedvr2_dir(ctx.project_root)

    def subprocess_env(self, ctx: EngineContext) -> dict[str, str]:
        root = seedvr2_dir(ctx.project_root)
        return {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "VIDEOX_SEEDVR_ROOT": str(root),
            "VIDEOX_SEEDVR_PROGRESS": "1",
        }

    def parse_log_line(self, line: str) -> LogInsight | None:
        return parse_seedvr2_line(line)

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        params = job.params
        gpus = ctx.gpus if ctx.gpus is not None else query_gpus()
        devices = job.devices if job.devices is not None else default_devices(gpus)
        if not devices:
            raise VideoXError("没有可用的 CUDA 设备")
        batch_size = params.get("batch_size")
        if batch_size is None:
            batch_size = batch_size_for_devices(devices, gpus)
        batch_size = int(batch_size)
        if batch_size < 1 or (batch_size - 1) % 4 != 0:
            raise VideoXError(f"batch_size 必须是 4n+1，当前是 {batch_size}")

        resolution = int(params.get("resolution", DEFAULT_RESOLUTION))
        temporal_overlap = int(params.get("temporal_overlap", DEFAULT_TEMPORAL_OVERLAP))
        prepend_frames = int(params.get("prepend_frames", DEFAULT_PREPEND_FRAMES))
        chunk_size = int(params.get("chunk_size", 0))
        blocks_to_swap = int(params.get("blocks_to_swap", 0))
        dit_model = str(params.get("dit_model", DEFAULT_DIT_MODEL))
        color_correction = str(params.get("color_correction", DEFAULT_COLOR_CORRECTION))
        seed = int(params.get("seed", DEFAULT_SEED))
        attention_mode = str(params.get("attention_mode", DEFAULT_ATTENTION_MODE))
        vae_encode_tiled = bool(params.get("vae_encode_tiled", True))
        vae_decode_tiled = bool(params.get("vae_decode_tiled", True))
        uniform_batch_size = bool(params.get("uniform_batch_size", True))

        command = [
            ctx.python_executable,
            "-u",
            "-m",
            "videox.engines.seedvr2.progress_entry",
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--output_format",
            "mp4",
            "--model_dir",
            str(ctx.model_dir),
            "--dit_model",
            dit_model,
            "--resolution",
            str(resolution),
            "--batch_size",
            str(batch_size),
            "--temporal_overlap",
            str(temporal_overlap),
            "--prepend_frames",
            str(prepend_frames),
            "--color_correction",
            color_correction,
            "--seed",
            str(seed),
            "--cuda_device",
            ",".join(str(device) for device in devices),
            "--video_backend",
            "ffmpeg",
            "--attention_mode",
            attention_mode,
        ]
        if uniform_batch_size:
            command.append("--uniform_batch_size")
        if vae_encode_tiled:
            command.append("--vae_encode_tiled")
        if vae_decode_tiled:
            command.append("--vae_decode_tiled")
        if chunk_size > 0:
            command.extend(["--chunk_size", str(chunk_size)])
        if blocks_to_swap > 0:
            offload = str(params.get("dit_offload_device") or "cpu")
            command.extend(["--blocks_to_swap", str(blocks_to_swap), "--dit_offload_device", offload])
        return command
