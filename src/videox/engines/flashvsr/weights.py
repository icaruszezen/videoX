from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file, missing_files, present_files
from videox.engines.flashvsr.presets import DEFAULT_MODE, WEIGHT_FILES, WEIGHT_REPO, weights_for


def ensure_weights(directory: Path, names: tuple[str, ...] = WEIGHT_FILES) -> None:
    for name in missing_files(directory, names):
        download_file(WEIGHT_REPO + name, directory / name)


def weight_lists(directory: Path) -> tuple[list[str], list[str]]:
    # Runs fetch only their mode's files, so only the default mode's count as missing;
    # Full downloads Wan2.1_VAE.pth the first time it is used.
    return present_files(directory, WEIGHT_FILES), missing_files(directory, weights_for(DEFAULT_MODE))
