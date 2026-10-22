from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from videox.engines.base import EngineContext
from videox.engines.registry import EngineRegistry
from videox.errors import VideoXError
from videox.jobs.models import EventType, JobRecord, JobSpec, JobStatus, ProgressEvent
from videox.jobs.store import JobStore
from videox.hardware.gpus import query_gpus
from videox.media.ffmpeg import MediaInfo, probe_media, remux_audio, transmux_video
from videox.media.kinds import IMAGE_SUFFIXES, resolve_media
from videox.paths import ensure_tool_path, model_dir


# The last line of a Python traceback, e.g. "torch.OutOfMemoryError: CUDA out of memory".
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b")

# Containers the remux step can write H.264/HEVC into. Anything else (clip.v2,
# movie.4k, .webm) would only fail after the whole restore has run.
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".avi"}


def video_only_path(output: Path) -> Path:
    return output.with_name(output.stem + ".video-only.mp4")


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    # Engines start in their own session, so the group also holds SeedVR2's GPU workers.
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return


def _same_file(a: Path, b: Path) -> bool:
    if a == b:
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


class JobRunner:
    def __init__(self, registry: EngineRegistry, store: JobStore, project_root: Path) -> None:
        self.registry = registry
        self.store = store
        self.project_root = project_root
        self._cancelled: set[str] = set()
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        # Jobs share the GPUs, so background jobs run one at a time in submit order.
        self._pending: queue.Queue[tuple[str, Callable[[str], None] | None]] = queue.Queue()
        self._worker: threading.Thread | None = None

    def submit(
        self,
        spec: JobSpec,
        *,
        background: bool,
        echo: Callable[[str], None] | None = None,
    ) -> JobRecord:
        spec = self._prepare_spec(spec)
        record = self.store.create(spec)
        self.store.publish(
            ProgressEvent(job_id=record.id, type=EventType.queued, message="已排队", stage="queued")
        )
        if background:
            with self._lock:
                if self._worker is None:
                    self._worker = threading.Thread(target=self._drain, daemon=True)
                    self._worker.start()
            self._pending.put((record.id, echo))
            return self.store.get(record.id) or record
        self.execute(record.id, echo=echo)
        return self.store.get(record.id) or record

    def _drain(self) -> None:
        while True:
            job_id, echo = self._pending.get()
            try:
                self.execute(job_id, echo=echo)
            except Exception:
                # execute records its own failures; never let one job stop the queue.
                pass

    def cancel(self, job_id: str) -> JobRecord:
        if self.store.get(job_id) is None:
            raise KeyError(job_id)
        with self._lock:
            self._cancelled.add(job_id)
            process = self._procs.get(job_id)
        if process is not None and process.poll() is None:
            kill_process_tree(process.pid)
        current = self.store.get(job_id)
        if current is not None and current.status == JobStatus.queued:
            self.store.finish(job_id, JobStatus.cancelled, "已取消")
        updated = self.store.get(job_id)
        if updated is None:
            raise KeyError(job_id)
        return updated

    def execute(self, job_id: str, echo: Callable[[str], None] | None = None) -> None:
        ensure_tool_path()
        record = self.store.get(job_id)
        if record is None or job_id in self._cancelled:
            if record is not None:
                self.store.finish(job_id, JobStatus.cancelled, "已取消")
            return
        if not self.store.update(job_id, status=JobStatus.running, stage="prepare"):
            return
        spec = record.spec
        image = spec.media == "image"
        png_sequence = not image and str(spec.params.get("output_format") or "mp4") == "png"
        self._emit(
            ProgressEvent(
                job_id=job_id,
                type=EventType.stage,
                stage="prepare",
                message="正在读取图片" if image else "正在读取视频",
            ),
            echo,
        )
        try:
            engine = self.registry.get(spec.engine_id)
            source = Path(spec.input_path)
            dest = Path(spec.output_path)
            if image:
                if dest.exists() and dest.is_dir():
                    raise VideoXError(f"图片输出需要是文件: {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                engine_output = dest
                info = None
            elif png_sequence:
                if dest.exists() and not dest.is_dir():
                    raise VideoXError(f"PNG 序列的输出需要是目录: {dest}")
                dest.mkdir(parents=True, exist_ok=True)
                engine_output = dest
                info = None
            else:
                info = probe_media(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                engine_output = video_only_path(dest)
            ctx = self._engine_context(engine)
            ctx.model_dir.mkdir(parents=True, exist_ok=True)
            engine_spec = spec.model_copy(update={"output_path": str(engine_output)})
            command = engine.build_command(engine_spec, ctx)
            workdir = engine.working_directory(ctx)
            if workdir is not None and not workdir.is_dir():
                raise VideoXError(f"引擎目录不存在: {workdir}。请先运行 videox setup。")
            if job_id in self._cancelled:
                self.store.finish(job_id, JobStatus.cancelled, "已取消")
                return
            self._run_engine(job_id, engine, command, ctx, echo)
            if job_id in self._cancelled:
                self.store.finish(job_id, JobStatus.cancelled, "已取消")
                return
            if image:
                if not dest.is_file():
                    raise VideoXError(f"引擎没有写出图片: {dest}")
            elif png_sequence:
                if not any(path.suffix.lower() == ".png" for path in dest.rglob("*") if path.is_file()):
                    raise VideoXError(f"引擎没有写出图片序列: {dest}")
            else:
                if not engine_output.is_file():
                    raise VideoXError(f"引擎没有写出视频: {engine_output}")
                assert info is not None
                self._emit(
                    ProgressEvent(job_id=job_id, type=EventType.stage, stage="audio", message="正在封装音轨"),
                    echo,
                )
                self._finish_video(engine, spec, info, source, engine_output, dest)
        except VideoXError as exc:
            if job_id in self._cancelled:
                self.store.finish(job_id, JobStatus.cancelled, "已取消")
                return
            self.store.finish(job_id, JobStatus.failed, str(exc))
            return
        except Exception as exc:
            if job_id in self._cancelled:
                self.store.finish(job_id, JobStatus.cancelled, "已取消")
                return
            self.store.finish(job_id, JobStatus.failed, f"处理失败: {exc}")
            return
        if job_id in self._cancelled:
            self.store.finish(job_id, JobStatus.cancelled, "已取消")
            return
        self.store.finish(job_id, JobStatus.completed, f"完成: {spec.output_path}")

    def _finish_video(
        self,
        engine,
        spec: JobSpec,
        info: MediaInfo,
        source: Path,
        engine_output: Path,
        dest: Path,
    ) -> None:
        try:
            rendered = probe_media(engine_output)
        except VideoXError:
            rendered = None
        hevc = rendered is not None and rendered.video_codec == "hevc"
        audio_offset = 0.0
        offset_for = getattr(engine, "audio_offset", None)
        if callable(offset_for):
            audio_offset = offset_for(spec.params, info)
        keep_audio = info.has_audio
        if audio_offset is None or (info.duration and audio_offset >= info.duration):
            # The engine skipped frames but the offset is unknown, or nothing of
            # the soundtrack is left: a silent result beats a misaligned one.
            keep_audio = False
        try:
            if keep_audio:
                remux_audio(
                    engine_output,
                    source,
                    dest,
                    audio_offset=audio_offset or 0.0,
                    max_duration=rendered.duration if rendered else None,
                    hevc=hevc,
                )
                engine_output.unlink(missing_ok=True)
            else:
                transmux_video(engine_output, dest, hevc=hevc)
        except VideoXError as exc:
            raise VideoXError(f"{exc}\n修复好的无声视频保留在: {engine_output}") from exc
        if not dest.is_file():
            raise VideoXError(f"成片不存在: {dest}")

    def _engine_context(self, engine) -> EngineContext:
        model_directory = getattr(engine, "model_directory", None)
        weights = (
            model_directory(self.project_root)
            if callable(model_directory)
            else model_dir(self.project_root)
        )
        ctx = EngineContext(
            project_root=self.project_root,
            python_executable=sys.executable,
            model_dir=weights,
            gpus=None,
        )
        resolve_python = getattr(engine, "resolve_python", None)
        if callable(resolve_python):
            ctx.python_executable = resolve_python(ctx)
        return ctx

    def _prepare_spec(self, spec: JobSpec) -> JobSpec:
        engine = self.registry.get(spec.engine_id)
        manifest = engine.manifest()
        input_path = Path(spec.input_path).expanduser().resolve()
        if not input_path.is_file():
            raise VideoXError(f"输入文件不存在: {input_path}")
        media = resolve_media(spec.media, input_path)
        if media not in manifest.media:
            kind = "图片" if media == "image" else "视频"
            raise VideoXError(f"{manifest.name} 不支持{kind}")
        params = dict(spec.params)
        if "output_format" not in (manifest.params_schema.get("properties") or {}):
            # Only engines that offer the choice can write a PNG sequence.
            params.pop("output_format", None)
        png_sequence = media == "video" and str(params.get("output_format") or "mp4") == "png"
        output_path = Path(spec.output_path).expanduser().resolve()
        if media == "image":
            if output_path.suffix.lower() not in IMAGE_SUFFIXES:
                output_path = output_path.with_suffix(".png")
        elif not png_sequence and output_path.suffix.lower() not in VIDEO_SUFFIXES:
            output_path = output_path.with_name(output_path.name + ".mp4")
        if _same_file(output_path, input_path):
            raise VideoXError("输出不能和输入是同一个文件")
        if media == "video" and not png_sequence and _same_file(video_only_path(output_path), input_path):
            raise VideoXError("输出文件名会覆盖输入视频，请换一个输出路径")
        prepared = spec.model_copy(
            update={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "media": media,
                "params": params,
            }
        )
        # Build the command once now so bad parameters, a missing engine install or
        # a machine without a GPU are reported when the job is submitted, not after
        # it has been queued.
        ctx = self._engine_context(engine)
        ctx.gpus = query_gpus()
        engine.build_command(prepared, ctx)
        return prepared

    def preview_directory(self, job_id: str) -> Path:
        return self.project_root / "output" / ".previews" / job_id

    def _emit(self, event: ProgressEvent, echo: Callable[[str], None] | None) -> None:
        self.store.publish(event)
        if echo is None or not event.message:
            return
        if event.type in {EventType.log, EventType.queued}:
            echo(event.message)
            return
        if event.type == EventType.stage and event.stage in {"prepare", "audio"}:
            echo(event.message)

    def _run_engine(
        self,
        job_id: str,
        engine,
        command: list[str],
        ctx: EngineContext,
        echo: Callable[[str], None] | None,
    ) -> None:
        self._emit(
            ProgressEvent(
                job_id=job_id,
                type=EventType.log,
                message=" ".join(command),
                stage="restore",
            ),
            echo,
        )
        env = os.environ.copy()
        env.update(engine.subprocess_env(ctx))
        preview_dir = self.preview_directory(job_id)
        preview_dir.mkdir(parents=True, exist_ok=True)
        env["VIDEOX_PREVIEW_DIR"] = str(preview_dir)
        # GPU indices shown in the UI come from nvidia-smi, which counts in PCI bus
        # order. CUDA counts fastest-first unless told otherwise.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        creationflags = 0
        session: dict[str, bool] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            session["start_new_session"] = True
        # Engines whose progress depends on earlier lines hand out a parser per job.
        new_parser = getattr(engine, "new_log_parser", None)
        parse_line = new_parser() if callable(new_parser) else engine.parse_log_line
        process = subprocess.Popen(
            command,
            cwd=str(engine.working_directory(ctx) or self.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            **session,
        )
        with self._lock:
            self._procs[job_id] = process
            cancelled = job_id in self._cancelled
        if cancelled:
            # cancel() ran between the last check and the registration above.
            kill_process_tree(process.pid)
        assert process.stdout is not None
        failure: str | None = None
        try:
            for line in process.stdout:
                text = line.rstrip("\r\n")
                if not text:
                    continue
                if text.startswith("错误:"):
                    failure = text.removeprefix("错误:").strip()
                elif _EXCEPTION_LINE.match(text):
                    failure = text.strip()
                insight = parse_line(text)
                if insight is not None and insight.preview_frame is not None:
                    self._emit(
                        ProgressEvent(
                            job_id=job_id,
                            type=EventType.preview,
                            preview_frame=insight.preview_frame,
                        ),
                        echo,
                    )
                    continue
                self._emit(ProgressEvent(job_id=job_id, type=EventType.log, message=text), echo)
                if insight is None:
                    continue
                if insight.stage:
                    self._emit(
                        ProgressEvent(
                            job_id=job_id,
                            type=EventType.stage,
                            stage=insight.stage,
                        ),
                        echo,
                    )
                if insight.progress is not None:
                    self._emit(
                        ProgressEvent(
                            job_id=job_id,
                            type=EventType.progress,
                            stage=insight.stage,
                            progress=insight.progress,
                        ),
                        echo,
                    )
            return_code = process.wait()
        finally:
            with self._lock:
                self._procs.pop(job_id, None)
        if job_id in self._cancelled:
            raise VideoXError("已取消")
        if return_code != 0:
            raise VideoXError(_failure_message(return_code, failure))


def _failure_message(return_code: int, failure: str | None) -> str:
    message = f"引擎退出码 {return_code}"
    if failure:
        return f"{message}：{failure[:500]}"
    if return_code in (-9, 137):
        return f"{message}（进程被系统强制结束，常见原因是内存不足）"
    return message
