from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from videox.hardware.gpus import query_gpus
from videox.paths import model_dir, project_root, read_expected_rev, seedvr2_dir, tool_executable
from videox.engines.seedvr2.presets import REQUIRED_MODELS


class GpuReport(BaseModel):
    index: int
    name: str
    memory_mib: int


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
    ready: bool = False
    warnings: list[str] = Field(default_factory=list)


def _torch_status() -> tuple[str | None, bool]:
    try:
        import torch
    except Exception:
        return None, False
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
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
    lines.append("可以运行: " + ("是" if report.ready else "否"))
    for warning in report.warnings:
        lines.append(f"注意: {warning}")
    return "\n".join(lines)
