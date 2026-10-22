from __future__ import annotations

import json

from videox.engines.base import LogInsight

# Each phase owns a slice of the bar, so the bar does not fill up while the
# weights download and then drop back to zero for the restore itself.
_PHASE_RANGES = {
    "download": (0.0, 0.05),
    "load": (0.05, 0.05),
    "face": (0.10, 0.45),
    "restore": (0.10, 0.90),
}
# GFPGAN restores the faces first and then reports the background as "restore".
_RESTORE_AFTER_FACES = (0.55, 0.45)


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


def parse_engine_line(line: str, *, after_faces: bool = False) -> LogInsight | None:
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
    ratio = None
    if total > 0:
        if phase == "restore" and after_faces:
            base, span = _RESTORE_AFTER_FACES
        else:
            base, span = _PHASE_RANGES.get(phase, (0.0, 1.0))
        ratio = base + span * min(current / total, 1.0)
    return LogInsight(stage=phase, progress=ratio)


class FaceThenRestoreParser:
    """parse_engine_line for one job of an engine that has a face phase before restore."""

    def __init__(self) -> None:
        self._after_faces = False

    def __call__(self, line: str) -> LogInsight | None:
        insight = parse_engine_line(line, after_faces=self._after_faces)
        if insight is not None and insight.stage == "face":
            self._after_faces = True
        return insight
