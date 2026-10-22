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

    setup_parser = sub.add_parser("setup", help="准备虚拟环境、ffmpeg 和修复引擎")
    setup_parser.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)

    sub.add_parser("doctor", help="检查本机能否运行")

    restore_parser = sub.add_parser("restore", help="修复一段视频或图片")
    restore_parser.add_argument("input")
    restore_parser.add_argument("-o", "--output")
    restore_parser.add_argument("--resolution", type=int, help="短边分辨率，只对 SeedVR2 生效")
    restore_parser.add_argument("--devices", help="CUDA 设备号，例如 0,1")
    restore_parser.add_argument("--batch-size", type=int, help="批大小，只对 SeedVR2 生效")
    restore_parser.add_argument(
        "--engine",
        default="seedvr2",
        help="引擎：seedvr2、flashvsr、swiftvr、realbasicvsr；图片还可以用 realesrgan、swinir、scunet、gfpgan",
    )
    restore_parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="名称=值",
        help="引擎参数，可以重复，例如 --param model=real_x2_m --param tile=256；名称见桌面端或 /engines",
    )

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
    output = (
        Path(args.output)
        if args.output
        else default_restore_output(root, source, args.engine, media)
    )
    params = restore_params(args.engine, args.resolution, args.batch_size)
    params.update(engine_params(args.engine, args.param))
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


def default_restore_output(root: Path, source: Path, engine_id: str, media: str) -> Path:
    suffix = ".png" if media == "image" else ".mp4"
    return root / "output" / f"{source.stem}.{engine_id}{suffix}"


def restore_params(engine_id: str, resolution: int | None, batch_size: int | None) -> dict:
    if engine_id != "seedvr2":
        return {}
    params: dict = {}
    if resolution is not None:
        params["resolution"] = resolution
    if batch_size is not None:
        params["batch_size"] = batch_size
    return params


def engine_params(engine_id: str, items: list[str]) -> dict[str, object]:
    """Turn repeated ``--param name=value`` into values typed by the engine's schema."""
    if not items:
        return {}
    properties = default_registry().get(engine_id).manifest().params_schema.get("properties", {})
    params: dict[str, object] = {}
    for item in items:
        key, separator, text = item.partition("=")
        key = key.strip()
        if not separator or key not in properties:
            raise VideoXError(f"{engine_id} 没有参数 {key or item}")
        kind = properties[key].get("type")
        try:
            if kind == "integer":
                params[key] = int(text)
            elif kind == "number":
                params[key] = float(text)
            elif kind == "boolean":
                if text.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(text)
                params[key] = text.lower() in {"true", "1"}
            else:
                params[key] = text
        except ValueError:
            raise VideoXError(f"参数 {key} 的值无法识别: {text}") from None
    return params


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))
