from __future__ import annotations

import os
from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.devices import chosen_device
from videox.engines.flashvsr.logs import parse_flashvsr_line
from videox.engines.flashvsr.presets import (
    DEFAULT_KV_RATIO,
    DEFAULT_LOCAL_RANGE,
    DEFAULT_MODE,
    DEFAULT_SCALE,
    DEFAULT_SEED,
    DEFAULT_SPARSE_RATIO,
    LOCAL_RANGE_CHOICES,
    MODE_CHOICES,
    SCALE_CHOICES,
    params_schema,
)
from videox.errors import VideoXError
from videox.jobs.models import EngineManifest, JobSpec
from videox.paths import flashvsr_dir, venv_python, weights_dir


class FlashVSREngine:
    id = "flashvsr"

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="FlashVSR",
            description="一步扩散超分，默认用 Tiny Long 流式处理长视频。比 SeedVR2 更快、更省显存，主要做 4 倍放大。只使用一张显卡。",
            capabilities=["upscale"],
            media=["video"],
            device_policy="single",
            params_schema=params_schema(),
        )

    def model_directory(self, root: Path) -> Path:
        return weights_dir("FlashVSR", root)

    def resolve_python(self, ctx: EngineContext) -> str:
        python = venv_python(flashvsr_dir(ctx.project_root))
        if not python.is_file():
            raise VideoXError("FlashVSR 还没有安装。请运行 videox setup。")
        return str(python)

    def working_directory(self, ctx: EngineContext) -> Path | None:
        return flashvsr_dir(ctx.project_root) / "examples" / "WanVSR"

    def subprocess_env(self, ctx: EngineContext) -> dict[str, str]:
        root = flashvsr_dir(ctx.project_root)
        wan = root / "examples" / "WanVSR"
        src = ctx.project_root / "src"
        current = os.environ.get("PYTHONPATH", "")
        parts = [str(src), str(wan), str(root)]
        if current:
            parts.append(current)
        return {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(parts),
            "VIDEOX_FLASHVSR_ROOT": str(root),
        }

    def parse_log_line(self, line: str) -> LogInsight | None:
        return parse_flashvsr_line(line)

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        params = job.params
        mode = _choice(_as_str(params.get("mode", DEFAULT_MODE), "模式"), MODE_CHOICES, "模式")
        scale = _choice(_as_str(params.get("scale", DEFAULT_SCALE), "放大倍数"), SCALE_CHOICES, "放大倍数")
        local_range = _choice(
            _as_str(params.get("local_range", DEFAULT_LOCAL_RANGE), "局部范围"),
            LOCAL_RANGE_CHOICES,
            "局部范围",
        )
        sparse_ratio = _as_float(params.get("sparse_ratio", DEFAULT_SPARSE_RATIO), "稀疏比", 0.5, 5)
        kv_ratio = _as_float(params.get("kv_ratio", DEFAULT_KV_RATIO), "KV 比", 1, 8)
        seed = _as_int(params.get("seed", DEFAULT_SEED), "随机种子")
        color_fix = _as_bool(params.get("color_fix", True), "色彩校正")
        device = chosen_device(job, ctx)
        command = [
            ctx.python_executable,
            "-u",
            "-m",
            "videox.engines.flashvsr.progress_entry",
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--model-dir",
            str(self.model_directory(ctx.project_root)),
            "--mode",
            mode,
            "--scale",
            scale,
            "--sparse-ratio",
            _format_float(sparse_ratio),
            "--local-range",
            local_range,
            "--kv-ratio",
            _format_float(kv_ratio),
            "--seed",
            str(seed),
            "--device",
            str(device),
        ]
        command.append("--color-fix" if color_fix else "--no-color-fix")
        return command


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoXError(f"{label} 需要是整数")
    if isinstance(value, float) and not value.is_integer():
        raise VideoXError(f"{label} 需要是整数")
    return int(value)


def _as_float(value: object, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoXError(f"{label} 需要是数字")
    number = float(value)
    if number < low or number > high:
        raise VideoXError(f"{label} 需要在 {low:g} 到 {high:g} 之间")
    return number


def _as_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise VideoXError(f"{label} 需要是布尔值")
    return value


def _as_str(value: object, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise VideoXError(f"{label} 需要是字符串")
    return str(value)


def _choice(value: str, choices: list[tuple[str, str]], label: str) -> str:
    allowed = {item for item, _label in choices}
    if value not in allowed:
        raise VideoXError(f"{label} 无法识别: {value}")
    return value


def _format_float(value: float) -> str:
    return format(value, "g")
