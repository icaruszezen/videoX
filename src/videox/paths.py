from __future__ import annotations

import os
import shutil
from pathlib import Path

SEEDVR2_REV_NAME = "seedvr2.rev"
SEEDVR2_DIR_NAME = "seedvr2"


def project_root() -> Path:
    override = os.environ.get("VIDEOX_ROOT")
    if override:
        return Path(override).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    cwd = Path.cwd()
    for path in (cwd, *cwd.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "videox").exists():
            return path
    return candidate


def seedvr2_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "third_party" / SEEDVR2_DIR_NAME


def seedvr2_rev_file(root: Path | None = None) -> Path:
    return (root or project_root()) / "third_party" / SEEDVR2_REV_NAME


def model_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "SEEDVR2"


def weights_dir(name: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / name


def third_party(name: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "third_party" / name


def flashvsr_dir(root: Path | None = None) -> Path:
    return third_party("flashvsr", root)


def swiftvr_dir(root: Path | None = None) -> Path:
    return third_party("swiftvr", root)


def flashvsr_rev_file(root: Path | None = None) -> Path:
    return third_party("flashvsr.rev", root)


def swiftvr_rev_file(root: Path | None = None) -> Path:
    return third_party("swiftvr.rev", root)


def venv_python(engine_dir: Path) -> Path:
    if os.name == "nt":
        return engine_dir / ".venv" / "Scripts" / "python.exe"
    return engine_dir / ".venv" / "bin" / "python"


def deps_marker(engine_dir: Path) -> Path:
    """Written by setup: "ready" when the engine's imports worked, otherwise "missing"."""
    return engine_dir / ".videox-deps"


def engine_venv_ready(engine_dir: Path) -> bool:
    if not venv_python(engine_dir).is_file():
        return False
    try:
        return deps_marker(engine_dir).read_text(encoding="utf-8").strip() != "missing"
    except OSError:
        # Installed before setup checked the imports.
        return True


def read_rev_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return text.split()[0]
    raise ValueError(f"{path} 没有提交号")


def read_expected_rev(root: Path | None = None) -> str:
    return read_rev_file(seedvr2_rev_file(root))


def bundled_ffmpeg() -> Path | None:
    base = project_root() / "tools" / "ffmpeg"
    if not base.exists():
        return None
    matches = sorted(base.rglob("ffmpeg.exe"))
    return matches[0] if matches else None


def ensure_tool_path() -> None:
    ffmpeg = bundled_ffmpeg()
    if ffmpeg is None:
        return
    bindir = str(ffmpeg.parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if bindir not in parts:
        os.environ["PATH"] = bindir + os.pathsep + current


def tool_executable(name: str) -> str | None:
    ensure_tool_path()
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return shutil.which(name + ".exe")
    return None
