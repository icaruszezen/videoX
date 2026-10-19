from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from videox.hardware.gpus import GpuInfo
from videox.jobs.models import EngineManifest, JobSpec


@dataclass(frozen=True)
class LogInsight:
    stage: str | None = None
    progress: float | None = None


@dataclass
class EngineContext:
    project_root: Path
    python_executable: str
    model_dir: Path
    gpus: list[GpuInfo] | None = None


class RestorationEngine(Protocol):
    id: str

    def manifest(self) -> EngineManifest: ...

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]: ...

    def parse_log_line(self, line: str) -> LogInsight | None: ...

    def working_directory(self, ctx: EngineContext) -> Path | None: ...

    def subprocess_env(self, ctx: EngineContext) -> dict[str, str]: ...
