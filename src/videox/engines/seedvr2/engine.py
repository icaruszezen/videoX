from __future__ import annotations

from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.seedvr2.logs import SeedVR2LogParser, parse_seedvr2_line
from videox.engines.seedvr2.presets import (
    ATTENTION_CHOICES,
    COLOR_CORRECTION_CHOICES,
    COMPILE_BACKEND_CHOICES,
    COMPILE_MODE_CHOICES,
    DEFAULT_ATTENTION_MODE,
    DEFAULT_BATCH_SIZE_IMAGE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLOR_CORRECTION,
    DEFAULT_DIT_MODEL,
    DEFAULT_PREPEND_FRAMES,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    DEFAULT_TEMPORAL_OVERLAP,
    MAX_SEED,
    OUTPUT_FORMAT_CHOICES,
    max_blocks_for_model,
    params_schema,
)
from videox.errors import VideoXError
from videox.hardware.gpus import batch_size_for_devices, default_devices, query_gpus
from videox.hardware.memory import physical_memory_bytes
from videox.jobs.models import EngineManifest, JobSpec
from videox.media.ffmpeg import MediaInfo
from videox.media.frames import probe_video
from videox.paths import seedvr2_dir

# Containers upstream's inference_cli accepts as video input (VIDEO_EXTENSIONS).
SEEDVR2_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class SeedVR2Engine:
    id = "seedvr2"

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="SeedVR2",
            description="一步扩散修复，支持视频和图片。双卡时视频按帧分段，每张卡各加载一份所选模型，但整段结果要先放在内存里，内存放不下的长视频只用一张卡；图片只用一张卡。",
            capabilities=["upscale"],
            media=["video", "image"],
            device_policy="multi",
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

    def new_log_parser(self) -> SeedVR2LogParser:
        return SeedVR2LogParser()

    def audio_offset(self, params: dict, info: MediaInfo) -> float | None:
        """Seconds of source audio to skip so it lines up with the restored picture."""
        skip = params.get("skip_first_frames", 0)
        if isinstance(skip, bool) or not isinstance(skip, (int, float)) or skip <= 0:
            return 0.0
        if not info.fps:
            return None
        return float(skip) / info.fps

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        params = job.params
        video = job.media != "image"
        source = Path(job.input_path)
        if video and source.suffix.lower() not in SEEDVR2_VIDEO_SUFFIXES:
            allowed = "/".join(sorted(suffix[1:] for suffix in SEEDVR2_VIDEO_SUFFIXES))
            raise VideoXError(f"SeedVR2 只能读取 {allowed} 格式的视频，请先转成 MP4 或换用其他引擎: {source.name}")
        gpus = ctx.gpus if ctx.gpus is not None else query_gpus()
        devices = job.devices if job.devices is not None else default_devices(gpus)
        if not devices:
            raise VideoXError("没有可用的 CUDA 设备")
        note: str | None = None
        if not video:
            # Upstream splits a one-frame image into fewer pieces than GPUs and then
            # waits forever for the missing worker; one card is enough for one image.
            devices = list(devices)[:1]
        else:
            cap = params.get("load_cap", 0)
            if isinstance(cap, int) and not isinstance(cap, bool) and 0 < cap < len(devices):
                # A GPU with no frames to process crashes its worker.
                devices = list(devices)[:cap]
            if len(devices) > 1:
                devices, note = _devices_that_fit_in_memory(source, params, list(devices), job.devices is not None)

        if video and "batch_size" not in params:
            batch_size = batch_size_for_devices(devices, gpus)
        elif video:
            batch_size = _as_int(params.get("batch_size"), "批大小")
        else:
            batch_size = (
                DEFAULT_BATCH_SIZE_IMAGE
                if "batch_size" not in params
                else _as_int(params.get("batch_size"), "批大小")
            )
        if batch_size < 1 or (batch_size - 1) % 4 != 0:
            raise VideoXError(f"batch_size 必须是 4n+1，当前是 {batch_size}")

        resolution = _as_int(params.get("resolution", DEFAULT_RESOLUTION), "短边分辨率")
        if resolution < 1:
            raise VideoXError("短边分辨率需要是正整数")
        dit_model = _as_str(params.get("dit_model", DEFAULT_DIT_MODEL), "扩散模型") or DEFAULT_DIT_MODEL
        color_correction = _choice(
            _as_str(params.get("color_correction", DEFAULT_COLOR_CORRECTION), "色彩校正"),
            COLOR_CORRECTION_CHOICES,
            "色彩校正",
        )
        seed = _as_int(params.get("seed", DEFAULT_SEED), "随机种子")
        if not 0 <= seed <= MAX_SEED:
            raise VideoXError(f"随机种子需要在 0 到 {MAX_SEED} 之间")
        attention_mode = _choice(
            _as_str(params.get("attention_mode", DEFAULT_ATTENTION_MODE), "注意力模式"),
            ATTENTION_CHOICES,
            "注意力模式",
        )
        max_resolution = _as_int(params.get("max_resolution", 0), "长边上限")
        if max_resolution < 0:
            raise VideoXError("长边上限不能小于 0")
        input_noise = _as_unit(params.get("input_noise_scale", 0), "输入噪声")
        latent_noise = _as_unit(params.get("latent_noise_scale", 0), "潜空间噪声")
        vae_encode_tiled = _as_bool(params.get("vae_encode_tiled", True), "VAE 编码分块")
        vae_decode_tiled = _as_bool(params.get("vae_decode_tiled", True), "VAE 解码分块")
        encode_tile_size = _as_int(params.get("vae_encode_tile_size", 1024), "编码块大小")
        encode_tile_overlap = _as_int(params.get("vae_encode_tile_overlap", 128), "编码块重叠")
        decode_tile_size = _as_int(params.get("vae_decode_tile_size", 1024), "解码块大小")
        decode_tile_overlap = _as_int(params.get("vae_decode_tile_overlap", 128), "解码块重叠")
        blocks_to_swap = _as_int(params.get("blocks_to_swap", 0), "换出块数")
        if blocks_to_swap < 0:
            raise VideoXError("换出块数不能小于 0")
        block_limit = max_blocks_for_model(dit_model)
        if blocks_to_swap > block_limit:
            raise VideoXError(f"换出块数最多 {block_limit}，当前是 {blocks_to_swap}")
        swap_io = _as_bool(params.get("swap_io_components", False), "交换输入输出组件")
        dit_offload = _offload(_as_str(params.get("dit_offload_device", "none"), "DiT 卸载设备"))
        if (blocks_to_swap > 0 or swap_io) and dit_offload == "none":
            dit_offload = "cpu"
        vae_offload = _offload(_as_str(params.get("vae_offload_device", "none"), "VAE 卸载设备"))
        tensor_offload = _offload(_as_str(params.get("tensor_offload_device", "cpu"), "张量卸载设备"))
        compile_dit = _as_bool(params.get("compile_dit", False), "编译 DiT")
        compile_vae = _as_bool(params.get("compile_vae", False), "编译 VAE")
        compile_backend = _choice(
            _as_str(params.get("compile_backend", "inductor"), "编译后端"),
            COMPILE_BACKEND_CHOICES,
            "编译后端",
        )
        compile_mode = _choice(
            _as_str(params.get("compile_mode", "default"), "编译模式"),
            COMPILE_MODE_CHOICES,
            "编译模式",
        )
        compile_fullgraph = _as_bool(params.get("compile_fullgraph", False), "整图编译")
        compile_dynamic = _as_bool(params.get("compile_dynamic", False), "动态形状")

        output_format = "png"
        use_10bit = False
        if video:
            output_format = _choice(
                _as_str(params.get("output_format", "mp4"), "输出格式"),
                OUTPUT_FORMAT_CHOICES,
                "输出格式",
            )
            use_10bit = _as_bool(params.get("use_10bit", False), "10-bit 输出")
            if use_10bit and output_format != "mp4":
                raise VideoXError("10-bit 只适用于 MP4")

        command = [
            ctx.python_executable,
            "-u",
            "-m",
            "videox.engines.seedvr2.progress_entry",
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--output_format",
            output_format,
            "--model_dir",
            str(ctx.model_dir),
            "--dit_model",
            dit_model,
            "--resolution",
            str(resolution),
            "--batch_size",
            str(batch_size),
        ]
        if video:
            temporal_overlap = _as_int(params.get("temporal_overlap", DEFAULT_TEMPORAL_OVERLAP), "时间重叠")
            prepend_frames = _as_int(params.get("prepend_frames", DEFAULT_PREPEND_FRAMES), "前置帧")
            if temporal_overlap < 0 or prepend_frames < 0:
                raise VideoXError("时间重叠和前置帧不能小于 0")
            command.extend(
                [
                    "--temporal_overlap",
                    str(temporal_overlap),
                    "--prepend_frames",
                    str(prepend_frames),
                ]
            )
        if note:
            command.extend(["--videox_note", note])
        command.extend(
            [
                "--color_correction",
                color_correction,
                "--seed",
                str(seed),
                "--cuda_device",
                ",".join(str(device) for device in devices),
            ]
        )
        if video and output_format == "mp4":
            command.extend(["--video_backend", "ffmpeg"])
        command.extend(["--attention_mode", attention_mode])
        if video and _as_bool(params.get("uniform_batch_size", True), "统一批大小"):
            command.append("--uniform_batch_size")
        if vae_encode_tiled:
            _check_tiles(encode_tile_size, encode_tile_overlap, "编码")
            command.append("--vae_encode_tiled")
            if encode_tile_size != 1024:
                command.extend(["--vae_encode_tile_size", str(encode_tile_size)])
            if encode_tile_overlap != 128:
                command.extend(["--vae_encode_tile_overlap", str(encode_tile_overlap)])
        if vae_decode_tiled:
            _check_tiles(decode_tile_size, decode_tile_overlap, "解码")
            command.append("--vae_decode_tiled")
            if decode_tile_size != 1024:
                command.extend(["--vae_decode_tile_size", str(decode_tile_size)])
            if decode_tile_overlap != 128:
                command.extend(["--vae_decode_tile_overlap", str(decode_tile_overlap)])
        if video:
            chunk_size = _as_int(params.get("chunk_size", DEFAULT_CHUNK_SIZE), "分块大小")
            skip_first = _as_int(params.get("skip_first_frames", 0), "跳过开头")
            load_cap = _as_int(params.get("load_cap", 0), "最多处理帧数")
            if chunk_size < 0 or skip_first < 0 or load_cap < 0:
                raise VideoXError("分块大小、跳过开头和最多处理帧数不能小于 0")
            if chunk_size > 0:
                command.extend(["--chunk_size", str(chunk_size)])
                if _as_bool(params.get("cache_models", True), "分块间保留模型"):
                    command.extend(["--cache_dit", "--cache_vae"])
            if skip_first > 0:
                command.extend(["--skip_first_frames", str(skip_first)])
            if load_cap > 0:
                command.extend(["--load_cap", str(load_cap)])
            if use_10bit:
                command.append("--10bit")
        if max_resolution > 0:
            command.extend(["--max_resolution", str(max_resolution)])
        if input_noise != 0:
            command.extend(["--input_noise_scale", _format_float(input_noise)])
        if latent_noise != 0:
            command.extend(["--latent_noise_scale", _format_float(latent_noise)])
        if blocks_to_swap > 0:
            command.extend(["--blocks_to_swap", str(blocks_to_swap)])
        if swap_io:
            command.append("--swap_io_components")
        if dit_offload != "none":
            command.extend(["--dit_offload_device", dit_offload])
        if vae_offload != "none":
            command.extend(["--vae_offload_device", vae_offload])
        if tensor_offload != "cpu":
            command.extend(["--tensor_offload_device", tensor_offload])
        if compile_dit:
            command.append("--compile_dit")
        if compile_vae:
            command.append("--compile_vae")
        if compile_dit or compile_vae:
            command.extend(["--compile_backend", compile_backend])
            if compile_backend == "inductor":
                # torch.compile only accepts a mode for Inductor.
                command.extend(["--compile_mode", compile_mode])
            if compile_fullgraph:
                command.append("--compile_fullgraph")
            if compile_dynamic:
                command.append("--compile_dynamic")
        return command


def _check_tiles(size: int, overlap: int, label: str) -> None:
    if size < 1:
        raise VideoXError(f"{label}块大小需要是正整数")
    if overlap < 0:
        raise VideoXError(f"{label}块重叠不能小于 0")
    if overlap >= size:
        raise VideoXError(f"{label}块重叠（{overlap}）必须小于{label}块大小（{size}）")



def _devices_that_fit_in_memory(
    source: Path, params: dict, devices: list[int], explicit: bool
) -> tuple[list[int], str | None]:
    """Several GPUs keep the whole restored video in RAM until the end; long clips need one.

    Upstream's multi-GPU path collects every worker's frames as float32 and joins
    them before writing, while one GPU writes each chunk as it finishes.
    """
    need = _whole_output_bytes(source, params)
    ram = physical_memory_bytes()
    # The parent holds the workers' frames and a joined float32 copy; leave half the RAM free.
    if need is None or ram is None or need * 2 <= ram // 2:
        return devices, None
    message = (
        f"SeedVR2 用多张显卡时要把整段修复结果留在内存里，这段视频约需 {need * 2 / 1e9:.0f} GB，"
        f"本机内存 {ram / 1e9:.0f} GB。"
    )
    if explicit:
        raise VideoXError(
            message + "请只选一张显卡（单卡按“分块大小”边处理边写出），或用“跳过开头”和“最多处理帧数”分段处理。"
        )
    return devices[:1], message + "这次只用第一张显卡，按分块边处理边写出。"


def _whole_output_bytes(source: Path, params: dict) -> int | None:
    """float32 RGB bytes of the whole restored clip, from the probe and the size settings."""
    try:
        probe = probe_video(source)
    except (VideoXError, OSError):
        return None
    frames = probe.frame_count
    skip = params.get("skip_first_frames", 0)
    if isinstance(skip, int) and not isinstance(skip, bool) and skip > 0:
        frames -= skip
    cap = params.get("load_cap", 0)
    if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
        frames = min(frames, cap)
    resolution = params.get("resolution", DEFAULT_RESOLUTION)
    longest = params.get("max_resolution", 0)
    if frames <= 0 or not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 1:
        return None
    short_side, long_side = sorted((probe.width, probe.height))
    if short_side < 1:
        return None
    out_short, out_long = resolution, resolution * long_side / short_side
    if isinstance(longest, int) and not isinstance(longest, bool) and 0 < longest < out_long:
        out_short, out_long = out_short * longest / out_long, longest
    return int(frames * out_short * out_long * 3 * 4)


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoXError(f"{label} 需要是整数")
    if isinstance(value, float) and not value.is_integer():
        raise VideoXError(f"{label} 需要是整数")
    return int(value)


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise VideoXError(f"{label} 需要是布尔值")
    return value


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise VideoXError(f"{label} 需要是字符串")
    return value


def _as_unit(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoXError(f"{label} 需要是数字")
    number = float(value)
    if number < 0 or number > 1:
        raise VideoXError(f"{label} 需要在 0 到 1 之间")
    return number


def _choice(value: str, choices: list[tuple[str, str]], label: str) -> str:
    allowed = {item for item, _label in choices}
    if value not in allowed:
        raise VideoXError(f"{label} 无法识别: {value}")
    return value


def _offload(value: str) -> str:
    if value in {"none", "cpu"} or value.startswith("cuda:"):
        if value.startswith("cuda:"):
            index = value.removeprefix("cuda:")
            if not index.isdigit():
                raise VideoXError(f"卸载设备无法识别: {value}")
        return value
    raise VideoXError(f"卸载设备无法识别: {value}")


def _format_float(value: float) -> str:
    return format(value, "g")
