from __future__ import annotations

import json
import re

from videox.engines.base import LogInsight

_CHUNK = re.compile(r"Chunk\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_PHASE_RANGES = (
    ("encoding", 0.00, 0.20),
    ("upscaling", 0.20, 0.25),
    ("decoding", 0.45, 0.50),
    ("post", 0.95, 0.05),
)


def _phase_range(phase: str) -> tuple[float, float]:
    lowered = phase.lower()
    for key, base, span in _PHASE_RANGES:
        if key in lowered:
            return base, span
    return 0.0, 1.0


def _stage_name(phase: str) -> str:
    lowered = phase.lower()
    for key in ("encoding", "upscaling", "decoding", "post"):
        if key in lowered:
            return key
    return phase or "restore"


def parse_seedvr2_line(line: str) -> LogInsight | None:
    """Parse one line without per-job state; see SeedVR2LogParser for the bar."""
    parsed = _parse(line)
    if parsed is None:
        return None
    insight, _worker, _workers = parsed
    return insight


class SeedVR2LogParser:
    """Turns SeedVR2's per-phase steps into one bar that only moves forward.

    Each phase reports its own 0..N steps, streaming runs repeat all four
    phases per chunk, and two GPU workers report at the same time. Every
    worker gets an equal share of the bar, filled chunk by chunk.
    """

    def __init__(self) -> None:
        self._done: dict[int, float] = {}
        self._workers = 1
        self._progress = 0.0

    def __call__(self, line: str) -> LogInsight | None:
        parsed = _parse(line)
        if parsed is None:
            return None
        insight, worker, workers = parsed
        if insight.progress is None or worker is None:
            return insight
        self._workers = max(self._workers, workers)
        self._done[worker] = max(self._done.get(worker, 0.0), insight.progress)
        overall = sum(self._done.values()) / self._workers
        self._progress = max(self._progress, min(overall, 1.0))
        return LogInsight(stage=insight.stage, progress=self._progress)


def _parse(line: str) -> tuple[LogInsight, int | None, int] | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith("{") and "videox_preview" in text:
        try:
            frame = int(json.loads(text)["videox_preview"]["frame"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if frame < 1:
            return None
        return LogInsight(preview_frame=frame), None, 1
    if text.startswith("{") and "videox_progress" in text:
        try:
            payload = json.loads(text)["videox_progress"]
            phase = str(payload.get("phase") or "")
            current = float(payload.get("current") or 0)
            total = float(payload.get("total") or 0)
            worker = int(payload.get("worker") or 0)
            workers = max(1, int(payload.get("workers") or 1))
            chunks = max(1, int(payload.get("chunks") or 1))
            chunk = min(max(1, int(payload.get("chunk") or 1)), chunks)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            return None
        progress = None
        if total > 0:
            base, span = _phase_range(phase)
            within = base + span * min(current / total, 1.0)
            progress = (chunk - 1 + within) / chunks
        return LogInsight(stage=_stage_name(phase), progress=progress), worker, workers
    if _CHUNK.search(text):
        # The JSON progress lines already account for chunks.
        return LogInsight(stage="chunk"), None, 1
    if "Output saved to" in text or "All upscaling processes completed" in text:
        return LogInsight(stage="save", progress=1.0), None, 1
    return None
