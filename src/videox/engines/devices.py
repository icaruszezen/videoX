from __future__ import annotations

from videox.engines.base import EngineContext
from videox.errors import VideoXError
from videox.hardware.gpus import default_devices, query_gpus
from videox.jobs.models import JobSpec


def chosen_device(job: JobSpec, ctx: EngineContext) -> int:
    """Use the first selected GPU. These engines do not split a video across cards."""
    gpus = ctx.gpus if ctx.gpus is not None else query_gpus()
    devices = job.devices if job.devices is not None else default_devices(gpus)
    if not devices:
        raise VideoXError("没有可用的 CUDA 设备")
    return int(devices[0])
