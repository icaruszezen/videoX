from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file, missing_files, present_files
from videox.engines.gfpgan.presets import DEFAULT_VERSION, MODELS, model_spec


def ensure_model(directory: Path, version: str) -> list[Path]:
    """Download the version's missing files and return their paths: restorer, then detector."""
    paths: list[Path] = []
    for weight in model_spec(version).files:
        path = directory / weight.name
        download_file(weight.url, path)
        paths.append(path)
    return paths


def known_weights() -> tuple[str, ...]:
    names: list[str] = []
    for spec in MODELS.values():
        for weight in spec.files:
            if weight.name not in names:
                names.append(weight.name)
    return tuple(names)


def weight_lists(directory: Path) -> tuple[list[str], list[str]]:
    """Every known file already on disk, and the default version's files still missing.

    Other versions download on first use, so they never count as missing. The
    Real-ESRGAN background weights live in the RealESRGAN folder and are not
    listed here.
    """
    default = tuple(weight.name for weight in MODELS[DEFAULT_VERSION].files)
    return present_files(directory, known_weights()), missing_files(directory, default)
