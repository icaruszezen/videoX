from fastapi.testclient import TestClient

from videox.engines.registry import EngineRegistry
from videox.engines.seedvr2.progress_entry import (
    PreviewCursor,
    parse_preview_args,
    postprocess_batch_index,
)
from videox.jobs.models import JobSpec
from videox.jobs.runner import JobRunner
from videox.jobs.store import JobStore
from videox.server import create_app


def test_parse_preview_args_reads_the_input_and_offsets():
    argv = [
        "progress_entry.py",
        r"D:\clips\in.mp4",
        "--output",
        r"D:\clips\out.mp4",
        "--prepend_frames",
        "4",
        "--skip_first_frames",
        "2",
    ]
    assert parse_preview_args(argv) == (r"D:\clips\in.mp4", 2, 4)


def test_cursor_skips_prepended_frames_then_chunk_context():
    cursor = PreviewCursor(segment_start=0, prepend=4, skip_first=0)
    assert cursor.locate(3, 3) is None
    assert cursor.locate(4, 3) == (0, 1)
    assert cursor.locate(9, 3) == (5, 6)

    cursor.advance(10, 3, prepend_already_removed=False)
    assert cursor.segment_start == 6
    assert cursor.calls == 1
    assert cursor.locate(2, 3) is None
    assert cursor.locate(3, 3) == (6, 7)
    assert cursor.locate(8, 3) == (11, 12)


def test_cursor_numbers_frames_from_the_segment_start():
    skipped = PreviewCursor(segment_start=10, prepend=4, skip_first=10)
    assert skipped.locate(4, 0) == (10, 1)

    worker = PreviewCursor(segment_start=100, prepend=0, skip_first=0)
    assert worker.locate(0, 0) == (100, 101)


def test_cursor_does_not_subtract_prepend_twice_when_it_was_removed():
    cursor = PreviewCursor(0, 4, 0)
    cursor.advance(6, 3, prepend_already_removed=True)
    assert cursor.segment_start == 6
    assert cursor.locate(0, 3) is None
    assert cursor.locate(3, 3) == (6, 7)


def test_batch_index_skips_the_alpha_pass():
    assert postprocess_batch_index(1, 4, 4) == 0
    assert postprocess_batch_index(4, 4, 4) == 3
    assert postprocess_batch_index(4, 8, 4) is None
    assert postprocess_batch_index(5, 8, 4) == 0
    assert postprocess_batch_index(8, 8, 4) == 3
    assert postprocess_batch_index(0, 4, 4) is None


def test_preview_routes_serve_the_job_pair(tmp_path):
    store = JobStore()
    runner = JobRunner(EngineRegistry(), store, tmp_path)
    record = store.create(
        JobSpec(input_path=str(tmp_path / "in.mp4"), output_path=str(tmp_path / "out.mp4"))
    )
    directory = runner.preview_directory(record.id)
    directory.mkdir(parents=True)
    (directory / "output.jpg").write_bytes(b"output-jpeg")
    client = TestClient(create_app(runner))

    output = client.get(f"/jobs/{record.id}/preview/output")
    assert output.status_code == 200
    assert output.content == b"output-jpeg"
    assert output.headers["cache-control"] == "no-store"

    missing = client.get(f"/jobs/{record.id}/preview/source")
    assert missing.status_code == 404
    unknown = client.get("/jobs/missing/preview/output")
    assert unknown.status_code == 404
