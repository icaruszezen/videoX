"""Common shape for still-image engines that run in the main environment.

These engines keep their network code in this repository, download weights on
first use, and run one image per process on one GPU. Subclasses describe their
schema and turn resolved parameters into arguments for their progress entry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from videox.engines.base import EngineContext, LogInsight
from videox.engines.devices import chosen_device
from videox.engines.fields import resolve_params
from videox.engines.progress import parse_engine_line
from videox.jobs.models import Capability, EngineManifest, JobSpec
from videox.paths import weights_dir


class MainImageEngine:
    id: str
    name: str
    description: str
    capabilities: ClassVar[list[Capability]] = ["upscale"]
    weights_folder: str
    entry_module: str

    def params_schema(self) -> dict:
        raise NotImplementedError

    def entry_args(self, values: dict[str, object]) -> list[str]:
        """Arguments after ``--model-dir``, built from validated parameters."""
        raise NotImplementedError

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name=self.name,
            description=self.description,
            capabilities=list(self.capabilities),
            media=["image"],
            device_policy="single",
            params_schema=self.params_schema(),
        )

    def model_directory(self, root: Path) -> Path:
        return weights_dir(self.weights_folder, root)

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
        return parse_engine_line(line)

    def resolved_params(self, job: JobSpec) -> dict[str, object]:
        return resolve_params(self.params_schema(), job.params, job.media)

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        values = self.resolved_params(job)
        device = chosen_device(job, ctx)
        return [
            ctx.python_executable,
            "-u",
            "-m",
            self.entry_module,
            str(Path(job.input_path).expanduser().resolve()),
            "--output",
            str(Path(job.output_path).expanduser().resolve()),
            "--model-dir",
            str(self.model_directory(ctx.project_root)),
            *self.entry_args(values),
            "--device",
            str(device),
        ]


def format_number(value: object) -> str:
    return format(float(value), "g") if isinstance(value, float) else str(value)
