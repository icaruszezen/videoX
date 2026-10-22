from __future__ import annotations

import json

from videox.engines.base import LogInsight


def emit_progress(phase: str, current: float, total: float) -> None:
    print(
        json.dumps(
            {"videox_progress": {"phase": phase, "current": current, "total": total}},
            ensure_ascii=False,
        ),
        flush=True,
    )


def emit_preview(frame: int) -> None:
    if frame < 1:
        return
    print(json.dumps({"videox_preview": {"frame": frame}}, ensure_ascii=False), flush=True)


def parse_engine_line(line: str) -> LogInsight | None:
    text = line.strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    preview = payload.get("videox_preview")
    if isinstance(preview, dict):
        try:
            frame = int(preview["frame"])
        except (KeyError, TypeError, ValueError):
            return None
        if frame < 1:
            return None
        return LogInsight(preview_frame=frame)
    progress = payload.get("videox_progress")
    if not isinstance(progress, dict):
        return None
    phase = str(progress.get("phase") or "restore")
    try:
        current = float(progress.get("current") or 0)
        total = float(progress.get("total") or 0)
    except (TypeError, ValueError):
        return LogInsight(stage=phase)
    ratio = current / total if total > 0 else None
    return LogInsight(stage=phase, progress=ratio)
