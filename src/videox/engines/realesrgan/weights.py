from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file, missing_files, present_files
from videox.engines.realesrgan.presets import (
    DEFAULT_DENOISE_STRENGTH,
    DEFAULT_MODEL,
    MODELS,
    required_weights,
)


def ensure_model(
    directory: Path,
    model_id: str,
    *,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
) -> list[Path]:
    """Download the model's missing files and return their paths, main checkpoint first."""
    paths: list[Path] = []
    for weight in required_weights(model_id, denoise_strength):
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
    """Every known file already on disk, and the default model's files still missing.

    Other models download on first use, so they never count as missing.
    """
    default = tuple(weight.name for weight in MODELS[DEFAULT_MODEL].files)
    return present_files(directory, known_weights()), missing_files(directory, default)
