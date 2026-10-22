from __future__ import annotations

import os
from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.devices import chosen_device
from videox.engines.swiftvr.logs import parse_swiftvr_line
from videox.engines.swiftvr.presets import (
    DEFAULT_CLIP_LEN,
    DEFAULT_DTYPE,
    DEFAULT_UPSCALE,
    DTYPE_CHOICES,
    params_schema,
)
from videox.errors import VideoXError
from videox.jobs.models import EngineManifest, JobSpec
from videox.paths import swiftvr_dir, venv_python, weights_dir


class SwiftVREngine:
    id = "swiftvr"

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="SwiftVR",
            description="一步生成式修复，面向实时超分。权重是研究许可：个人本地使用可以，商用需要作者书面授权。只使用一张显卡。",
            capabilities=["upscale"],
            media=["video"],
            device_policy="single",
            params_schema=params_schema(),
        )

    def model_directory(self, root: Path) -> Path:
        return weights_dir("SwiftVR", root)

    def resolve_python(self, ctx: EngineContext) -> str:
        python = venv_python(swiftvr_dir(ctx.project_root))
        if not python.is_file():
            raise VideoXError("SwiftVR 还没有安装。请运行 videox setup。")
        return str(python)

    def working_directory(self, ctx: EngineContext) -> Path | None:
        return swiftvr_dir(ctx.project_root)

    def subprocess_env(self, ctx: EngineContext) -> dict[str, str]:
        src = ctx.project_root / "src"
        current = os.environ.get("PYTHONPATH", "")
        parts = [str(src)]
        if current:
            parts.append(current)
        return {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(parts),
        }

    def parse_log_line(self, line: str) -> LogInsight | None:
        return parse_swiftvr_line(line)

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        params = job.params
        upscale = _as_int(params.get("upscale", DEFAULT_UPSCALE), "放大倍数", 1, 4)
        clip_len = _as_int(params.get("clip_len", DEFAULT_CLIP_LEN), "分块长度", 4, 64)
        if clip_len % 4 != 0:
            raise VideoXError(f"分块长度必须是 4 的倍数，当前是 {clip_len}")
        dtype = _choice(_as_str(params.get("dtype", DEFAULT_DTYPE), "精度"), DTYPE_CHOICES, "精度")
        device = chosen_device(job, ctx)
        return [
            ctx.python_executable,
            "-u",
            "-m",
            "videox.engines.swiftvr.progress_entry",
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--checkpoint",
            str(self.model_directory(ctx.project_root)),
            "--upscale",
            str(upscale),
            "--clip-len",
            str(clip_len),
            "--dtype",
            dtype,
            "--device",
            str(device),
        ]


def _as_int(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoXError(f"{label} 需要是整数")
    if isinstance(value, float) and not value.is_integer():
        raise VideoXError(f"{label} 需要是整数")
    number = int(value)
    if number < low or number > high:
        raise VideoXError(f"{label} 需要在 {low} 到 {high} 之间")
    return number


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise VideoXError(f"{label} 需要是字符串")
    return value


def _choice(value: str, choices: list[tuple[str, str]], label: str) -> str:
    allowed = {item for item, _label in choices}
    if value not in allowed:
        raise VideoXError(f"{label} 无法识别: {value}")
    return value
