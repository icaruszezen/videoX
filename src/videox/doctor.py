from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from videox.engines.flashvsr.weights import weight_lists as flash_weight_lists
from videox.engines.realbasicvsr.weights import weight_lists as realbasic_weight_lists
from videox.engines.seedvr2.presets import REQUIRED_MODELS
from videox.engines.swiftvr.weights import weight_lists as swift_weight_lists
from videox.hardware.gpus import query_gpus
from videox.paths import (
    engine_venv_ready,
    flashvsr_dir,
    flashvsr_rev_file,
    model_dir,
    project_root,
    read_expected_rev,
    read_rev_file,
    seedvr2_dir,
    swiftvr_dir,
    swiftvr_rev_file,
    tool_executable,
    venv_python,
    weights_dir,
)


class GpuReport(BaseModel):
    index: int
    name: str
    memory_mib: int


class EngineStatus(BaseModel):
    id: str
    name: str
    present: bool = False
    commit: str | None = None
    commit_expected: str | None = None
    venv_ready: bool = False
    model_dir: str
    models_present: list[str] = Field(default_factory=list)
    models_missing: list[str] = Field(default_factory=list)
    detail: str = ""


class SystemReport(BaseModel):
    python: str
    project_root: str
    torch: str | None = None
    cuda_available: bool = False
    ffmpeg: str | None = None
    ffprobe: str | None = None
    gpus: list[GpuReport] = Field(default_factory=list)
    seedvr2_present: bool = False
    seedvr2_commit: str | None = None
    seedvr2_commit_expected: str | None = None
    model_dir: str
    models_present: list[str] = Field(default_factory=list)
    models_missing: list[str] = Field(default_factory=list)
    engines: list[EngineStatus] = Field(default_factory=list)
    ready: bool = False
    warnings: list[str] = Field(default_factory=list)


def _torch_status() -> tuple[str | None, bool]:
    # Never call torch.cuda.is_available() here. It creates a CUDA context on
    # the display GPU, and the desktop polls /system while a job is running.
    try:
        import torch
    except Exception:
        return None, False
    cuda = bool(getattr(getattr(torch, "version", None), "cuda", None))
    return str(torch.__version__), cuda


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _expected(path: Path) -> str | None:
    try:
        return read_rev_file(path)
    except (FileNotFoundError, ValueError):
        return None


def _engine_statuses(root: Path) -> list[EngineStatus]:
    seed_checkout = seedvr2_dir(root)
    seed_weights = model_dir(root)
    seed_present_files = [name for name in REQUIRED_MODELS if (seed_weights / name).is_file()]
    seed_missing = [name for name in REQUIRED_MODELS if name not in seed_present_files]
    seed_expected = _expected(root / "third_party" / "seedvr2.rev")
    seed_commit = _git_head(seed_checkout)
    seed_present = (seed_checkout / "inference_cli.py").is_file()
    flash_root = flashvsr_dir(root)
    flash_weights = weights_dir("FlashVSR", root)
    flash_present_files, flash_missing = flash_weight_lists(flash_weights)
    flash_expected = _expected(flashvsr_rev_file(root))
    flash_commit = _git_head(flash_root)
    flash_present = (flash_root / "diffsynth").is_dir()
    flash_python = venv_python(flash_root)
    sparse_marker = (flash_root / ".videox-sparse").read_text(encoding="utf-8").strip() if (
        flash_root / ".videox-sparse"
    ).is_file() else ""
    if sparse_marker == "ready":
        flash_detail = "稀疏注意力已编译"
    elif sparse_marker == "missing":
        flash_detail = "缺少 Block-Sparse-Attention，FlashVSR 无法运行"
    elif flash_python.is_file():
        flash_detail = "尚未确认稀疏注意力"
    else:
        flash_detail = "使用独立虚拟环境"
    swift_root = swiftvr_dir(root)
    swift_weights = weights_dir("SwiftVR", root)
    swift_present_files, swift_missing = swift_weight_lists(swift_weights)
    real_weights = weights_dir("RealBasicVSR", root)
    real_present_files, real_missing = realbasic_weight_lists(real_weights)
    return [
        EngineStatus(
            id="seedvr2",
            name="SeedVR2",
            present=seed_present,
            commit=seed_commit,
            commit_expected=seed_expected,
            venv_ready=seed_present,
            model_dir=str(seed_weights),
            models_present=seed_present_files,
            models_missing=seed_missing,
            detail="使用主环境",
        ),
        EngineStatus(
            id="flashvsr",
            name="FlashVSR",
            present=flash_present,
            commit=flash_commit,
            commit_expected=flash_expected,
            venv_ready=engine_venv_ready(flash_root) and sparse_marker != "missing",
            model_dir=str(flash_weights),
            models_present=flash_present_files,
            models_missing=flash_missing,
            detail=flash_detail,
        ),
        EngineStatus(
            id="swiftvr",
            name="SwiftVR",
            present=(swift_root / "swiftvr").is_dir(),
            commit=_git_head(swift_root),
            commit_expected=_expected(swiftvr_rev_file(root)),
            venv_ready=engine_venv_ready(swift_root),
            model_dir=str(swift_weights),
            models_present=swift_present_files,
            models_missing=swift_missing,
            detail="使用独立虚拟环境。权重是研究许可，商用需要作者授权。",
        ),
        EngineStatus(
            id="realbasicvsr",
            name="RealBasicVSR",
            present=True,
            venv_ready=True,
            model_dir=str(real_weights),
            models_present=real_present_files,
            models_missing=real_missing,
            detail="使用主环境，固定 4 倍",
        ),
    ]


def _engine_warnings(engines: list[EngineStatus]) -> list[str]:
    warnings: list[str] = []
    for engine in engines:
        if engine.id == "seedvr2":
            continue
        if engine.id != "realbasicvsr" and not engine.present:
            warnings.append(f"{engine.name} 还没有检出，请运行 videox setup")
        elif engine.id != "realbasicvsr" and not engine.venv_ready:
            warnings.append(f"{engine.name} 的虚拟环境还没有装好，请运行 videox setup")
        elif (
            engine.commit_expected
            and engine.commit
            and engine.commit != engine.commit_expected
        ):
            warnings.append(f"{engine.name} 提交是 {engine.commit}，期望 {engine.commit_expected}")
        if engine.models_missing:
            warnings.append(
                f"{engine.name} 权重尚未下载，首次运行会自动拉取: " + ", ".join(engine.models_missing)
            )
        if engine.id == "flashvsr" and engine.detail.startswith("缺少 Block-Sparse"):
            warnings.append("FlashVSR 缺少 Block-Sparse-Attention，无法运行，请安装 CUDA Toolkit 后重新运行 videox setup")
    return warnings


def collect_report(root: Path | None = None) -> SystemReport:
    root = root or project_root()
    import sys

    gpus = [
        GpuReport(index=gpu.index, name=gpu.name, memory_mib=gpu.memory_mib)
        for gpu in query_gpus()
    ]
    checkout = seedvr2_dir(root)
    weights = model_dir(root)
    present = [name for name in REQUIRED_MODELS if (weights / name).is_file()]
    missing = [name for name in REQUIRED_MODELS if name not in present]
    expected = None
    try:
        expected = read_expected_rev(root)
    except (FileNotFoundError, ValueError):
        expected = None
    commit = _git_head(checkout)
    torch_version, cuda = _torch_status()
    engines = _engine_statuses(root)
    warnings: list[str] = []
    if not gpus:
        warnings.append("没有检测到 NVIDIA 显卡")
    if torch_version is None:
        warnings.append("当前环境没有安装 PyTorch")
    elif not cuda:
        warnings.append("PyTorch 看不到 CUDA")
    if tool_executable("ffmpeg") is None or tool_executable("ffprobe") is None:
        warnings.append("ffmpeg 或 ffprobe 不在 PATH 中")
    if not (checkout / "inference_cli.py").is_file():
        warnings.append("SeedVR2 还没有检出，请运行 videox setup")
    elif expected and commit != expected:
        warnings.append(f"SeedVR2 提交是 {commit}，期望 {expected}")
    if missing:
        warnings.append("模型权重尚未下载，首次 restore 会自动拉取: " + ", ".join(missing))
    warnings.extend(_engine_warnings(engines))
    ready = not any(
        item.startswith("没有检测")
        or item.startswith("当前环境")
        or item.startswith("PyTorch")
        or item.startswith("ffmpeg")
        or item.startswith("SeedVR2")
        for item in warnings
    )
    return SystemReport(
        python=sys.version.split()[0],
        project_root=str(root),
        torch=torch_version,
        cuda_available=cuda,
        ffmpeg=tool_executable("ffmpeg"),
        ffprobe=tool_executable("ffprobe"),
        gpus=gpus,
        seedvr2_present=(checkout / "inference_cli.py").is_file(),
        seedvr2_commit=commit,
        seedvr2_commit_expected=expected,
        model_dir=str(weights),
        models_present=present,
        models_missing=missing,
        engines=engines,
        ready=ready,
        warnings=warnings,
    )


def format_report(report: SystemReport) -> str:
    lines = [
        f"Python: {report.python}",
        f"项目目录: {report.project_root}",
        f"PyTorch: {report.torch or '未安装'}  CUDA: {'是' if report.cuda_available else '否'}",
        f"ffmpeg: {report.ffmpeg or '未找到'}",
        f"ffprobe: {report.ffprobe or '未找到'}",
        f"SeedVR2: {'已检出 ' + (report.seedvr2_commit or '') if report.seedvr2_present else '未检出'}",
        f"期望提交: {report.seedvr2_commit_expected or '未知'}",
        f"模型目录: {report.model_dir}",
    ]
    if report.gpus:
        lines.append("显卡:")
        for gpu in report.gpus:
            lines.append(f"  [{gpu.index}] {gpu.name}  {gpu.memory_mib} MiB")
    else:
        lines.append("显卡: 未检测到")
    if report.models_present:
        lines.append("已有模型: " + ", ".join(report.models_present))
    if report.models_missing:
        lines.append("缺少模型: " + ", ".join(report.models_missing))
    for engine in report.engines:
        state = "已就绪" if engine.present and engine.venv_ready and not engine.models_missing else "未就绪"
        lines.append(f"{engine.name}: {state}  {engine.detail}")
        if engine.models_missing:
            lines.append("  缺少权重: " + ", ".join(engine.models_missing))
    lines.append("可以运行: " + ("是" if report.ready else "否"))
    for warning in report.warnings:
        lines.append(f"注意: {warning}")
    return "\n".join(lines)
