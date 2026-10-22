from __future__ import annotations

import os
from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.devices import chosen_device
from videox.engines.realbasicvsr.logs import parse_realbasicvsr_line
from videox.engines.realbasicvsr.presets import DEFAULT_MAX_SEQ_LEN, params_schema
from videox.errors import VideoXError
from videox.jobs.models import EngineManifest, JobSpec
from videox.paths import weights_dir


class RealBasicVSREngine:
    id = "realbasicvsr"

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="RealBasicVSR",
            description="非扩散的循环超分，固定 4 倍，适合 720p 及以下的低清素材（1080p 原片会输出 8K，16GB 显卡放不下）。长视频按帧分段，速度通常更快，细节和人脸弱于扩散模型。只使用一张显卡。",
            capabilities=["upscale"],
            media=["video"],
            device_policy="single",
            params_schema=params_schema(),
        )

    def model_directory(self, root: Path) -> Path:
        return weights_dir("RealBasicVSR", root)

    def resolve_python(self, ctx: EngineContext) -> str:
        return ctx.python_executable

    def working_directory(self, ctx: EngineContext) -> Path | None:
        return None

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
        return parse_realbasicvsr_line(line)

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        max_seq_len = _as_int(job.params.get("max_seq_len", DEFAULT_MAX_SEQ_LEN), "一次处理帧数", 1, 64)
        device = chosen_device(job, ctx)
        return [
            ctx.python_executable,
            "-u",
            "-m",
            "videox.engines.realbasicvsr.progress_entry",
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--model-dir",
            str(self.model_directory(ctx.project_root)),
            "--max-seq-len",
            str(max_seq_len),
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
