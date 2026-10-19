import sys
from pathlib import Path

from videox.engines.base import EngineContext, LogInsight
from videox.jobs.models import EngineManifest, JobSpec, JobStatus
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.engines.registry import EngineRegistry
from videox.media.ffmpeg import MediaInfo


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
