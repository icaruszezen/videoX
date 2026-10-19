from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from videox.errors import VideoXError
from videox.paths import bundled_ffmpeg, project_root, read_expected_rev, seedvr2_dir, tool_executable

SEEDVR2_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
TORCH_INDEXES = (
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu130",
)


def _say(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
    if completed.returncode != 0:
        raise VideoXError(f"命令失败 ({completed.returncode}): {' '.join(command)}")


def _python_ok() -> None:
    if sys.version_info < (3, 12):
        raise VideoXError(f"需要 Python 3.12 或更新版本，当前是 {sys.version.split()[0]}")


def _nvidia_ok() -> None:
    if shutil.which("nvidia-smi") is None:
        raise VideoXError("找不到 nvidia-smi，请先安装 NVIDIA 驱动")
    completed = subprocess.run(
        ["nvidia-smi"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise VideoXError("nvidia-smi 无法运行，请检查 NVIDIA 驱动")


def ensure_ffmpeg(root: Path) -> None:
    if tool_executable("ffmpeg") and tool_executable("ffprobe"):
        return
    if bundled_ffmpeg() and tool_executable("ffmpeg") and tool_executable("ffprobe"):
        return
    destination = root / "tools" / "ffmpeg"
    destination.mkdir(parents=True, exist_ok=True)
    archive = root / ".work" / "ffmpeg.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    _say(f"正在下载 ffmpeg: {FFMPEG_URL}")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, archive)
    except OSError as exc:
        raise VideoXError(f"下载 ffmpeg 失败: {exc}") from exc
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink(missing_ok=True)
    from videox.paths import ensure_tool_path

    ensure_tool_path()
    if not tool_executable("ffmpeg") or not tool_executable("ffprobe"):
        raise VideoXError("ffmpeg 已解压，但没有找到 ffmpeg.exe / ffprobe.exe")


def ensure_seedvr2(root: Path) -> None:
    revision = read_expected_rev(root)
    destination = seedvr2_dir(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        if destination.exists():
            raise VideoXError(f"{destination} 已存在但不是 git 检出，请移走后重试")
        _say(f"正在克隆 SeedVR2 到 {destination}")
        _run(["git", "clone", SEEDVR2_URL, str(destination)])
    _say(f"正在检出 SeedVR2 {revision}")
    _run(["git", "fetch", "--tags", "origin"], cwd=destination)
    _run(["git", "checkout", "--detach", revision], cwd=destination)


def _cuda_ready(python: str) -> bool:
    completed = subprocess.run(
        [python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        check=False,
    )
    return completed.returncode == 0


def install_torch(python: str) -> None:
    last_error: VideoXError | None = None
    for index in TORCH_INDEXES:
        _say(f"正在安装 CUDA 版 PyTorch: {index}")
        try:
            _run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    index,
                ]
            )
        except VideoXError as exc:
            last_error = exc
            continue
        if _cuda_ready(python):
            return
        last_error = VideoXError(f"从 {index} 安装后 PyTorch 仍然看不到 CUDA")
    raise last_error or VideoXError("无法安装带 CUDA 的 PyTorch")


def install_seedvr_requirements(root: Path, python: str) -> None:
    requirements = seedvr2_dir(root) / "requirements.txt"
    if not requirements.is_file():
        raise VideoXError(f"缺少 {requirements}")
    filtered = root / ".work" / "seedvr2-requirements.txt"
    filtered.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        name = stripped.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        normalized = name.lower().replace("_", "-")
        if normalized in {"flash-attn", "sageattention", "apex"}:
            continue
        if normalized in {"torch", "torchvision", "torchaudio"}:
            continue
        kept.append(line)
    filtered.write_text("\n".join(kept) + "\n", encoding="utf-8")
    _say("正在安装 SeedVR2 依赖，不编译 flash-attn")
    _run([python, "-m", "pip", "install", "-r", str(filtered)])


def bootstrap_venv(root: Path) -> int:
    _python_ok()
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        _say(f"正在创建虚拟环境 {root / '.venv'}")
        _run([sys.executable, "-m", "venv", str(root / ".venv")])
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(venv_python), "-m", "pip", "install", "-e", f"{root}[dev]"])
    return subprocess.call([str(venv_python), "-m", "videox", "setup", "--in-venv"])


def setup_environment(root: Path | None = None, *, in_venv: bool = False) -> int:
    root = root or project_root()
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    inside = Path(sys.executable).resolve() == venv_python.resolve() if venv_python.exists() else False
    if not in_venv or not inside:
        if not inside:
            return bootstrap_venv(root)
    _python_ok()
    _nvidia_ok()
    ensure_ffmpeg(root)
    ensure_seedvr2(root)
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    _run([sys.executable, "-m", "pip", "install", "-e", f"{root}[dev]"])
    install_torch(sys.executable)
    install_seedvr_requirements(root, sys.executable)
    if not _cuda_ready(sys.executable):
        install_torch(sys.executable)
    _say("环境已准备好。接下来可以运行 videox doctor。")
    return 0
