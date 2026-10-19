"""Run SeedVR2 and emit one JSON progress object per line.

SeedVR2's phase functions accept a progress callback, but the CLI always
passes None. This entry wraps those functions before importing the CLI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PHASE_FUNCTIONS = (
    "encode_all_batches",
    "upscale_all_batches",
    "decode_all_batches",
    "postprocess_all_batches",
)


def _emit(current: int, total: int, frames: int, phase: str) -> None:
    print(
        json.dumps(
            {
                "videox_progress": {
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "frames": frames,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _wrap(function):
    if getattr(function, "_videox_wrapped", False):
        return function

    def wrapped(*args, **kwargs):
        original = kwargs.get("progress_callback")

        def callback(current, total, frames, phase):
            _emit(int(current), int(total), int(frames), str(phase))
            if original is not None:
                original(current, total, frames, phase)

        kwargs["progress_callback"] = callback
        return function(*args, **kwargs)

    wrapped._videox_wrapped = True
    return wrapped


def _install_wrappers() -> None:
    import src.core.generation_phases as phases

    for name in _PHASE_FUNCTIONS:
        setattr(phases, name, _wrap(getattr(phases, name)))


def main() -> int:
    root = Path(os.environ["VIDEOX_SEEDVR_ROOT"]).resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    saved_pythonpath = os.environ.get("PYTHONPATH", "")
    _install_wrappers()
    import inference_cli

    parts = [str(root)]
    if saved_pythonpath:
        parts.append(saved_pythonpath)
    # The upstream CLI joins PYTHONPATH with ':' which Windows does not split.
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    inference_cli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
