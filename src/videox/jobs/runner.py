from __future__ import annotations

import os
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
from videox.media.ffmpeg import probe_media, remux_audio, transmux_video
from videox.media.kinds import IMAGE_SUFFIXES, resolve_media
from videox.paths import ensure_tool_path, model_dir


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
    try:
        os.kill(pid, 15)
    except OSError:
        return


class JobRunner:
    def __init__(self, registry: EngineRegistry, store: JobStore, project_root: Path) -> None:
        self.registry = registry
        self.store = store
        self.project_root = project_root
        self._cancelled: set[str] = set()
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

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
            threading.Thread(
                target=self.execute,
                args=(record.id,),
                kwargs={"echo": echo},
                daemon=True,
            ).start()
            return self.store.get(record.id) or record
        self.execute(record.id, echo=echo)
        return self.store.get(record.id) or record

    def cancel(self, job_id: str) -> JobRecord:
        if self.store.get(job_id) is None:
            raise KeyError(job_id)
        self._cancelled.add(job_id)
        with self._lock:
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
            weights = model_dir(self.project_root)
            weights.mkdir(parents=True, exist_ok=True)
            ctx = EngineContext(
                project_root=self.project_root,
                python_executable=sys.executable,
                model_dir=weights,
                gpus=None,
            )
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
                if info.has_audio:
                    remux_audio(engine_output, source, dest)
                    engine_output.unlink(missing_ok=True)
                else:
                    transmux_video(engine_output, dest)
                if not dest.is_file():
                    raise VideoXError(f"成片不存在: {dest}")
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
        self.store.finish(job_id, JobStatus.completed, f"完成: {spec.output_path}")

    def _prepare_spec(self, spec: JobSpec) -> JobSpec:
        engine = self.registry.get(spec.engine_id)
        input_path = Path(spec.input_path).expanduser().resolve()
        if not input_path.is_file():
            raise VideoXError(f"输入文件不存在: {input_path}")
        media = resolve_media(spec.media, input_path)
        if media not in engine.manifest().media:
            kind = "图片" if media == "image" else "视频"
            raise VideoXError(f"{engine.manifest().name} 不支持{kind}")
        output_path = Path(spec.output_path).expanduser().resolve()
        if media == "image":
            if output_path.suffix.lower() not in IMAGE_SUFFIXES:
                output_path = output_path.with_suffix(".png")
        elif str(spec.params.get("output_format") or "mp4") != "png" and output_path.suffix == "":
            output_path = output_path.with_suffix(".mp4")
        return spec.model_copy(
            update={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "media": media,
            }
        )

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
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
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
        )
        with self._lock:
            self._procs[job_id] = process
        assert process.stdout is not None
        try:
            for line in process.stdout:
                text = line.rstrip("\r\n")
                if not text:
                    continue
                self._emit(ProgressEvent(job_id=job_id, type=EventType.log, message=text), echo)
                insight = engine.parse_log_line(text)
                if insight is None:
                    continue
                if insight.stage:
                    self._emit(
                        ProgressEvent(
                            job_id=job_id,
                            type=EventType.stage,
                            stage=insight.stage,
                            message=text,
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
                            message=text,
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
            raise VideoXError(f"引擎退出码 {return_code}")
