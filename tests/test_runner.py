import queue
import subprocess
import sys
import time
from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.engines.seedvr2.logs import parse_seedvr2_line
from videox.errors import VideoXError
from videox.jobs.models import EngineManifest, JobSpec, JobStatus
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.engines.registry import EngineRegistry
from videox.media.ffmpeg import MediaInfo
from videox.media.kinds import resolve_media
import pytest


class FakeEngine:
    id = "fake"

    def __init__(self, command: list[str]) -> None:
        self._command = command

    def manifest(self) -> EngineManifest:
        return EngineManifest(
            id=self.id,
            name="Fake",
            description="test",
            capabilities=["upscale"],
            media=["video", "image"],
            params_schema={"properties": {"output_format": {"type": "string"}}},
        )

    def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
        return [*self._command, job.output_path]

    def parse_log_line(self, line: str) -> LogInsight | None:
        return None

    def working_directory(self, ctx: EngineContext) -> Path | None:
        return None

    def subprocess_env(self, ctx: EngineContext) -> dict[str, str]:
        return {}


def _runner(tmp_path: Path, command: list[str]) -> JobRunner:
    registry = EngineRegistry()
    registry.register(FakeEngine(command))
    return JobRunner(registry, JobStore(), tmp_path)


def test_runner_remuxes_audio(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=True),
    )

    def fake_remux(video_only: Path, original: Path, output: Path, **options: object) -> None:
        assert video_only.read_bytes() == b"vid"
        assert original == source
        assert options["audio_offset"] == 0.0
        output.write_bytes(b"mux")

    monkeypatch.setattr("videox.jobs.runner.remux_audio", fake_remux)
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'vid')"
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(dest)),
        background=False,
    )
    assert record.status == JobStatus.completed
    assert dest.read_bytes() == b"mux"
    assert not (tmp_path / "out.video-only.mp4").exists()


def test_runner_moves_silent_video(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'vid')"
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(dest)),
        background=False,
    )
    assert record.status == JobStatus.completed
    assert dest.read_bytes() == b"vid"


def test_runner_writes_image_without_remux(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.png"
    source.write_bytes(b"in")
    dest = tmp_path / "out.png"

    def forbidden(path: Path, *args: object, **kwargs: object) -> None:
        raise AssertionError(path)

    monkeypatch.setattr("videox.jobs.runner.probe_media", forbidden)
    monkeypatch.setattr("videox.jobs.runner.remux_audio", forbidden)
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'img')"
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(
            engine_id="fake",
            media="image",
            input_path=str(source),
            output_path=str(dest),
        ),
        background=False,
    )
    assert record.status == JobStatus.completed
    assert record.spec.media == "image"
    assert dest.read_bytes() == b"img"


def test_runner_keeps_png_sequence_in_place(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    dest = tmp_path / "frames"

    def forbidden(path: Path, *args: object, **kwargs: object) -> None:
        raise AssertionError(path)

    monkeypatch.setattr("videox.jobs.runner.probe_media", forbidden)
    monkeypatch.setattr("videox.jobs.runner.remux_audio", forbidden)
    code = (
        "from pathlib import Path; import sys; "
        "dest = Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True); "
        "(dest / 'frame_000000.png').write_bytes(b'png')"
    )
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(
            engine_id="fake",
            input_path=str(source),
            output_path=str(dest),
            params={"output_format": "png"},
        ),
        background=False,
    )
    assert record.status == JobStatus.completed
    assert (dest / "frame_000000.png").read_bytes() == b"png"


def test_runner_rejects_image_request_for_video(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    runner = _runner(tmp_path, [sys.executable, "-c", "pass"])
    with pytest.raises(VideoXError, match="图片"):
        runner.submit(
            JobSpec(
                engine_id="fake",
                media="image",
                input_path=str(source),
                output_path=str(tmp_path / "out.png"),
            ),
            background=False,
        )


def test_resolve_media_follows_the_requested_kind(tmp_path: Path):
    assert resolve_media("video", tmp_path / "clip.mp4") == "video"
    assert resolve_media("image", tmp_path / "still.png") == "image"
    with pytest.raises(VideoXError, match="图片文件"):
        resolve_media("video", tmp_path / "still.webp")


def test_runner_publishes_preview_frames_outside_the_log(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )

    class ParsingEngine(FakeEngine):
        def parse_log_line(self, line: str) -> LogInsight | None:
            return parse_seedvr2_line(line)

    registry = EngineRegistry()
    registry.register(ParsingEngine([sys.executable, "-c", _PREVIEW_SCRIPT]))
    runner = JobRunner(registry, JobStore(), tmp_path)
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(dest)),
        background=False,
    )
    assert record.status == JobStatus.completed
    assert (tmp_path / "output" / ".previews" / record.id).is_dir()
    subscription = runner.store.subscribe(record.id)
    events = []
    while True:
        try:
            events.append(subscription.get_nowait())
        except queue.Empty:
            break
    previews = [event for event in events if event.type.value == "preview"]
    logged = [event.message for event in events if event.type.value == "log"]
    assert [event.preview_frame for event in previews] == [4]
    assert '{"videox_preview": {"frame": 4}}' not in logged
    assert "hello" in logged


_PREVIEW_SCRIPT = """
print("hello", flush=True)
print('{"videox_preview": {"frame": 4}}', flush=True)
from pathlib import Path
import sys
Path(sys.argv[1]).write_bytes(b"vid")
"""


def test_runner_reports_engine_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    runner = _runner(tmp_path, [sys.executable, "-c", "import sys; sys.exit(3)"])
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(tmp_path / "out.mp4")),
        background=False,
    )
    assert record.status == JobStatus.failed
    assert record.error is not None
    assert "3" in record.error


def test_runner_failure_names_the_engine_error(tmp_path: Path, monkeypatch):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    code = (
        "import sys; print('Traceback (most recent call last):'); "
        "print('torch.OutOfMemoryError: CUDA out of memory'); print('done'); sys.exit(1)"
    )
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(tmp_path / "out.mp4")),
        background=False,
    )
    assert record.error == "引擎退出码 1：torch.OutOfMemoryError: CUDA out of memory"

    code = "import sys; print('错误: 找不到模型文件'); sys.exit(1)"
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(
        JobSpec(engine_id="fake", input_path=str(source), output_path=str(tmp_path / "out2.mp4")),
        background=False,
    )
    assert record.error == "引擎退出码 1：找不到模型文件"


def _video_job(tmp_path: Path, output: str = "out.mp4", **changes: object) -> JobSpec:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"in")
    return JobSpec(engine_id="fake", input_path=str(source), output_path=str(tmp_path / output), **changes)


def test_runner_rejects_output_that_overwrites_the_input(tmp_path: Path):
    runner = _runner(tmp_path, [sys.executable, "-c", "pass"])
    job = _video_job(tmp_path)
    with pytest.raises(VideoXError, match="同一个文件"):
        runner.submit(job.model_copy(update={"output_path": job.input_path}), background=False)
    clash = tmp_path / "in.video-only.mp4"
    clash.write_bytes(b"in")
    with pytest.raises(VideoXError, match="覆盖输入"):
        runner.submit(
            JobSpec(engine_id="fake", input_path=str(clash), output_path=str(tmp_path / "in.mp4")),
            background=False,
        )


def test_runner_gives_unknown_video_suffixes_an_mp4_container(tmp_path: Path):
    runner = _runner(tmp_path, [sys.executable, "-c", "pass"])
    record = runner.store.create(runner._prepare_spec(_video_job(tmp_path, "clip.v2")))
    assert record.spec.output_path.endswith("clip.v2.mp4")
    record = runner.store.create(runner._prepare_spec(_video_job(tmp_path, "clip.mkv")))
    assert record.spec.output_path.endswith("clip.mkv")


def test_runner_ignores_png_output_for_engines_without_it(tmp_path: Path):
    class PlainEngine(FakeEngine):
        def manifest(self) -> EngineManifest:
            return super().manifest().model_copy(update={"params_schema": {}})

    registry = EngineRegistry()
    registry.register(PlainEngine([sys.executable, "-c", "pass"]))
    runner = JobRunner(registry, JobStore(), tmp_path)
    spec = runner._prepare_spec(_video_job(tmp_path, "frames", params={"output_format": "png"}))
    assert "output_format" not in spec.params
    assert spec.output_path.endswith("frames.mp4")


def test_runner_reports_bad_parameters_at_submit(tmp_path: Path):
    class StrictEngine(FakeEngine):
        def build_command(self, job: JobSpec, ctx: EngineContext) -> list[str]:
            raise VideoXError("批大小不对")

    registry = EngineRegistry()
    registry.register(StrictEngine([]))
    runner = JobRunner(registry, JobStore(), tmp_path)
    with pytest.raises(VideoXError, match="批大小"):
        runner.submit(_video_job(tmp_path), background=True)


def test_runner_aligns_audio_with_skipped_frames(tmp_path: Path, monkeypatch):
    calls = {}

    class SkippingEngine(FakeEngine):
        def audio_offset(self, params: dict, info: MediaInfo) -> float | None:
            return params["skip_first_frames"] / info.fps

    def fake_probe(path: Path) -> MediaInfo:
        if path.name.endswith(".video-only.mp4"):
            return MediaInfo(has_audio=False, duration=4.0, video_codec="hevc")
        return MediaInfo(has_audio=True, fps=25.0, duration=20.0)

    def fake_remux(video_only: Path, original: Path, output: Path, **options: object) -> None:
        calls.update(options)
        output.write_bytes(b"mux")

    monkeypatch.setattr("videox.jobs.runner.probe_media", fake_probe)
    monkeypatch.setattr("videox.jobs.runner.remux_audio", fake_remux)
    registry = EngineRegistry()
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'vid')"
    registry.register(SkippingEngine([sys.executable, "-c", code]))
    runner = JobRunner(registry, JobStore(), tmp_path)
    record = runner.submit(_video_job(tmp_path, params={"skip_first_frames": 250}), background=False)
    assert record.status == JobStatus.completed
    assert calls == {"audio_offset": 10.0, "max_duration": 4.0, "hevc": True}


def test_runner_runs_background_jobs_one_at_a_time(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    log = tmp_path / "order.log"
    code = (
        "import sys, time; from pathlib import Path; "
        f"log = open({str(log)!r}, 'a'); log.write('start\\n'); log.flush(); "
        "time.sleep(0.3); log.write('end\\n'); log.close(); "
        "Path(sys.argv[1]).write_bytes(b'vid')"
    )
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    first = runner.submit(_video_job(tmp_path, "a.mp4"), background=True)
    second = runner.submit(_video_job(tmp_path, "b.mp4"), background=True)
    assert runner.store.get(second.id).status == JobStatus.queued
    deadline = time.time() + 20
    while time.time() < deadline:
        states = {runner.store.get(job.id).status for job in (first, second)}
        if states == {JobStatus.completed}:
            break
        time.sleep(0.05)
    assert runner.store.get(second.id).status == JobStatus.completed
    assert log.read_text().split() == ["start", "end", "start", "end"]


def test_runner_puts_cuda_in_nvidia_smi_order(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    code = "import os, sys; from pathlib import Path; Path(sys.argv[1]).write_text(os.environ['CUDA_DEVICE_ORDER'])"
    runner = _runner(tmp_path, [sys.executable, "-c", code])
    record = runner.submit(_video_job(tmp_path), background=False)
    assert record.status == JobStatus.completed
    assert (tmp_path / "out.mp4").read_text() == "PCI_BUS_ID"


def test_cancel_before_the_engine_registers_still_stops_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "videox.jobs.runner.probe_media",
        lambda path: MediaInfo(has_audio=False),
    )
    runner = _runner(tmp_path, [sys.executable, "-c", "import time; time.sleep(30)"])
    original = subprocess.Popen

    def slow_popen(*args, **kwargs):
        # Cancel lands after execute's last check but before the process is registered.
        runner._cancelled.add(record_id[0])
        return original(*args, **kwargs)

    monkeypatch.setattr("videox.jobs.runner.subprocess.Popen", slow_popen)
    record_id: list[str] = []
    spec = runner._prepare_spec(_video_job(tmp_path))
    record = runner.store.create(spec)
    record_id.append(record.id)
    started = time.time()
    runner.execute(record.id)
    assert time.time() - started < 20
    assert runner.store.get(record.id).status == JobStatus.cancelled
