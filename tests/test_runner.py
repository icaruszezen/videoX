import queue
import sys
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
            params_schema={},
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

    def fake_remux(video_only: Path, original: Path, output: Path) -> None:
        assert video_only.read_bytes() == b"vid"
        assert original == source
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
