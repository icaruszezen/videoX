from videox.engines.seedvr2.logs import parse_seedvr2_line


def test_chunk_line_becomes_progress():
    insight = parse_seedvr2_line("[GPU 1] Chunk 2/4: 30 new + 3 context frames")
    assert insight is not None
    assert insight.stage == "chunk"
    assert insight.progress == 0.5


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
