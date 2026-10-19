from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_mib: int


def query_gpus() -> list[GpuInfo]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            memory = int(float(parts[-1]))
        except ValueError:
            continue
        name = ",".join(parts[1:-1]).strip()
        gpus.append(GpuInfo(index=index, name=name, memory_mib=memory))
    return gpus


def default_devices(gpus: list[GpuInfo]) -> list[int]:
    if len(gpus) >= 2:
        return [gpus[0].index, gpus[1].index]
    if len(gpus) == 1:
        return [gpus[0].index]
    return [0]


def batch_size_for_memory_mib(memory_mib: int) -> int:
    """12GB cards use 5 frames; about 16GB and above use 9."""
    if memory_mib >= 15 * 1024:
        return 9
    return 5


def batch_size_for_devices(devices: list[int], gpus: list[GpuInfo]) -> int:
    by_index = {gpu.index: gpu.memory_mib for gpu in gpus}
    memories = [by_index.get(device, 12 * 1024) for device in devices] or [12 * 1024]
    return batch_size_for_memory_mib(min(memories))
