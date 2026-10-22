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
        return LogInsight(preview_frame=frame)
    if text.startswith("{") and "videox_progress" in text:
        try:
            payload = json.loads(text)["videox_progress"]
            phase = str(payload.get("phase") or "")
            current = float(payload.get("current") or 0)
            total = float(payload.get("total") or 0)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        progress = None
        if total > 0:
            base, span = _phase_range(phase)
            progress = base + span * (current / total)
        return LogInsight(stage=_stage_name(phase), progress=progress)
    chunk = _CHUNK.search(text)
    if chunk:
        current = int(chunk.group(1))
        total = int(chunk.group(2))
        progress = current / total if total else None
        return LogInsight(stage="chunk", progress=progress)
    if "Output saved to" in text or "All upscaling processes completed" in text:
        return LogInsight(stage="save", progress=1.0)
    return None
