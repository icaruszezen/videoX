from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file
from videox.engines.realbasicvsr.presets import CHECKPOINT_NAME, CHECKPOINT_URL


def checkpoint_path(directory: Path) -> Path:
    return directory / CHECKPOINT_NAME


def ensure_checkpoint(directory: Path) -> Path:
    path = checkpoint_path(directory)
    download_file(CHECKPOINT_URL, path)
    return path


def weight_lists(directory: Path) -> tuple[list[str], list[str]]:
    path = checkpoint_path(directory)
    if path.is_file() and path.stat().st_size > 0:
        return [CHECKPOINT_NAME], []
    return [], [CHECKPOINT_NAME]
