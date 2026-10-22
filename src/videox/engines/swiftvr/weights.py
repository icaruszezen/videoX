from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file, missing_files, present_files
from videox.engines.swiftvr.presets import WEIGHT_FILES, WEIGHT_REPO


def ensure_weights(directory: Path) -> None:
    for name in missing_files(directory, WEIGHT_FILES):
        download_file(WEIGHT_REPO + name, directory / name)


def weight_lists(directory: Path) -> tuple[list[str], list[str]]:
    return present_files(directory, WEIGHT_FILES), missing_files(directory, WEIGHT_FILES)
