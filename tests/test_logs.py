import json

from videox.engines.seedvr2.logs import SeedVR2LogParser, parse_seedvr2_line


def test_chunk_line_only_sets_the_stage():
    insight = parse_seedvr2_line("[GPU 1] Chunk 2/4: 30 new + 3 context frames")
    assert insight is not None
    assert insight.stage == "chunk"
    assert insight.progress is None


def test_progress_json_maps_onto_phase_range():
    line = (
        '{"videox_progress": {"phase": "Phase 2: Upscaling", '
        '"current": 1, "total": 2, "frames": 1}}'
    )
    insight = parse_seedvr2_line(line)
    assert insight is not None
    assert insight.stage == "upscaling"
    assert insight.progress == 0.20 + 0.25 * 0.5
    assert insight.preview_frame is None


def test_preview_json_carries_the_frame_without_progress():
    insight = parse_seedvr2_line('{"videox_preview": {"frame": 12}}')
    assert insight is not None
    assert insight.preview_frame == 12
    assert insight.progress is None
    assert insight.stage is None
    assert parse_seedvr2_line('{"videox_preview": {"frame": 0}}') is None
    assert parse_seedvr2_line('{"videox_preview": {}}') is None


def _line(phase: str, current: int, total: int, **extra: int) -> str:
    payload = {"phase": phase, "current": current, "total": total, "frames": 1, **extra}
    return json.dumps({"videox_progress": payload})


def test_chunks_fill_the_bar_in_order():
    insight = parse_seedvr2_line(_line("Phase 1: Encoding", 1, 1, chunk=2, chunks=4))
    assert insight is not None
    assert insight.progress == (1 + 0.20) / 4


def test_parser_never_moves_backwards_across_chunks_and_workers():
    parse = SeedVR2LogParser()
    lines = [
        _line("Phase 3: Decoding", 1, 1, worker=0, workers=2, chunk=1, chunks=2),
        _line("Phase 1: Encoding", 1, 2, worker=1, workers=2, chunk=1, chunks=2),
        _line("Phase 1: Encoding", 1, 2, worker=0, workers=2, chunk=2, chunks=2),
        _line("Phase 4: Post-processing", 1, 1, worker=0, workers=2, chunk=2, chunks=2),
        _line("Phase 4: Post-processing", 1, 1, worker=1, workers=2, chunk=2, chunks=2),
    ]
    seen = []
    for line in lines:
        insight = parse(line)
        assert insight is not None and insight.progress is not None
        seen.append(insight.progress)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0
