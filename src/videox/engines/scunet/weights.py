from __future__ import annotations

from pathlib import Path

from videox.engines.downloads import download_file, missing_files, present_files
from videox.engines.scunet.presets import DEFAULT_MODEL, MODELS, model_spec


def known_files() -> tuple[str, ...]:
    names: list[str] = []
    for model in MODELS.values():
        names.extend(name for name in model.files if name not in names)
    return tuple(names)


def ensure_model(directory: Path, model_id: str) -> list[Path]:
    """Download the files of one model that are not on disk yet and return their paths."""
    model = model_spec(model_id)
    paths: list[Path] = []
    for name, url in zip(model.files, model.urls):
        path = directory / name
        download_file(url, path)
        paths.append(path)
    return paths


def weight_lists(directory: Path) -> tuple[list[str], list[str]]:
    """Every weight file already present, and the default model's files still missing.

    Other models download on first use, so they never count as missing.
    """
    return present_files(directory, known_files()), missing_files(directory, MODELS[DEFAULT_MODEL].files)
