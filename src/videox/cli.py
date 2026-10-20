from __future__ import annotations

import argparse
import sys
from pathlib import Path

from videox.doctor import collect_report, format_report
from videox.engines.registry import default_registry
from videox.errors import VideoXError
from videox.jobs.models import JobSpec, JobStatus
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.media.kinds import is_image_path
from videox.paths import ensure_tool_path, project_root
from videox.server import serve
from videox.setup_env import setup_environment


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _parse_devices(value: str | None) -> list[int] | None:
    if not value:
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise VideoXError(f"设备号无法解析: {value}") from exc


def run(argv: list[str] | None = None) -> int:
    _configure_stdio()
    ensure_tool_path()
    parser = argparse.ArgumentParser(prog="videox", description="本地视频修复")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_parser = sub.add_parser("setup", help="准备虚拟环境、ffmpeg 和 SeedVR2")
    setup_parser.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)

    sub.add_parser("doctor", help="检查本机能否运行")

    restore_parser = sub.add_parser("restore", help="用 SeedVR2 修复一段视频")
    restore_parser.add_argument("input")
    restore_parser.add_argument("-o", "--output")
    restore_parser.add_argument("--resolution", type=int)
    restore_parser.add_argument("--devices", help="CUDA 设备号，例如 0,1")
    restore_parser.add_argument("--batch-size", type=int)
    restore_parser.add_argument("--engine", default="seedvr2")

    serve_parser = sub.add_parser("serve", help="启动给以后界面使用的本地 API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            return setup_environment(in_venv=args.in_venv)
        if args.command == "doctor":
            report = collect_report()
            print(format_report(report))
            return 0 if report.ready else 1
        if args.command == "serve":
            serve(args.host, args.port)
            return 0
        if args.command == "restore":
            return _restore(args)
    except VideoXError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 2


def _restore(args: argparse.Namespace) -> int:
    root = project_root()
    source = Path(args.input)
    media = "image" if is_image_path(source) else "video"
    suffix = ".png" if media == "image" else ".mp4"
    output = Path(args.output) if args.output else root / "output" / f"{source.stem}.seedvr2{suffix}"
    params: dict = {}
    if args.resolution is not None:
        params["resolution"] = args.resolution
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    spec = JobSpec(
        engine_id=args.engine,
        input_path=str(source),
        output_path=str(output),
        media=media,
        devices=_parse_devices(args.devices),
        params=params,
    )
    runner = JobRunner(default_registry(), JobStore(), root)
    record = runner.submit(spec, background=False, echo=lambda line: print(line, flush=True))
    if record.status != JobStatus.completed:
        print(record.error or "修复失败", file=sys.stderr)
        return 1
    print(f"完成: {record.spec.output_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))
