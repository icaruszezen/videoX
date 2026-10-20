from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_STATUSES = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}


class EventType(str, Enum):
    queued = "queued"
    stage = "stage"
    log = "log"
    progress = "progress"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


MediaKind = Literal["video", "image"]


class JobSpec(BaseModel):
    engine_id: str = "seedvr2"
    input_path: str
    output_path: str
    media: MediaKind = "video"
    devices: list[int] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    id: str
    spec: JobSpec
    status: JobStatus
    created_at: str
    error: str | None = None
    progress: float | None = None
    stage: str | None = None


class ProgressEvent(BaseModel):
    job_id: str
    type: EventType
    message: str = ""
    progress: float | None = None
    stage: str | None = None


Capability = Literal["upscale", "interpolate", "face"]


class EngineManifest(BaseModel):
    id: str
    name: str
    description: str
    capabilities: list[Capability]
    media: list[MediaKind] = Field(default_factory=lambda: ["video"])
    params_schema: dict[str, Any]
