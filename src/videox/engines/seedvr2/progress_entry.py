"""Run SeedVR2 and emit one JSON progress object per line.

SeedVR2's phase functions accept a progress callback, but the CLI always
passes None. This entry wraps those functions before importing the CLI.
During post-processing it also writes the latest finished frame and the
matching source frame for the desktop preview.
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
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class PreviewCursor:
    """Tracks which source frame a tensor index corresponds to.

    The first post-process call in a process may start with prepended frames.
    Later calls start with temporal-overlap context from the previous chunk.
    Neither of those regions is part of the frames that get saved.
    """

    def __init__(self, segment_start: int, prepend: int, skip_first: int) -> None:
        self.segment_start = segment_start
        self.prepend = max(0, prepend)
        self.skip_first = max(0, skip_first)
        self.calls = 0

    def head(self, temporal_overlap: int) -> int:
        if self.calls == 0:
            return self.prepend
        return max(0, temporal_overlap)

    def locate(self, tensor_index: int, temporal_overlap: int) -> tuple[int, int] | None:
        if tensor_index < 0:
            return None
        head = self.head(temporal_overlap)
        if tensor_index < head:
            return None
        source = self.segment_start + tensor_index - head
        frame = source - self.skip_first + 1
        if frame < 1:
            return None
        return source, frame

    def advance(self, frame_count: int, temporal_overlap: int, *, prepend_already_removed: bool) -> None:
        if self.calls == 0 and prepend_already_removed:
            head = 0
        else:
            head = self.head(temporal_overlap)
        self.segment_start += max(0, frame_count - head)
        self.calls += 1


def parse_preview_args(argv: list[str]) -> tuple[str | None, int, int]:
    """Return the input path, skip_first_frames, and prepend_frames from a CLI argv."""
    input_path = None
    skip = 0
    prepend = 0
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--skip_first_frames" and index + 1 < len(argv):
            skip = _safe_int(argv[index + 1], skip)
            index += 2
            continue
        if token == "--prepend_frames" and index + 1 < len(argv):
            prepend = _safe_int(argv[index + 1], prepend)
            index += 2
            continue
        if not token.startswith("--") and input_path is None:
            input_path = token
        index += 1
    return input_path, skip, prepend


def postprocess_batch_index(current: int, total: int, batch_count: int) -> int | None:
    """Map a phase-4 progress step onto a color-corrected batch.

    RGBA runs an alpha pass over every batch before color correction. Those
    earlier steps still hold RGB in [-1, 1], so they are not preview frames.
    """
    if batch_count <= 0 or current <= 0 or total <= 0:
        return None
    if total > batch_count:
        if current <= total - batch_count:
            return None
        index = current - (total - batch_count) - 1
    else:
        index = current - 1
    if index < 0 or index >= batch_count:
        return None
    return index


class _PreviewState:
    def __init__(self) -> None:
        self.input_path: str | None = None
        self.cursor = PreviewCursor(0, 0, 0)
        self._capture = None

    def configure(self, input_path: str | None, segment_start: int, prepend: int, skip_first: int) -> None:
        self.input_path = input_path
        self.cursor = PreviewCursor(segment_start, prepend, skip_first)
        self.close()

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()


_state = _PreviewState()


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


def _emit_preview(frame: int) -> None:
    print(json.dumps({"videox_preview": {"frame": frame}}, ensure_ascii=False), flush=True)


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


def _wrap_postprocess(function):
    if getattr(function, "_videox_wrapped", False):
        return function

    def wrapped(*args, **kwargs):
        ctx = args[0] if args else kwargs.get("ctx")
        original = kwargs.get("progress_callback")
        overlap = _safe_int(_argument(args, kwargs, "temporal_overlap", 5, 0), 0)
        requested_prepend = _safe_int(_argument(args, kwargs, "prepend_frames", 4, 0), 0)

        def callback(current, total, frames, phase):
            _emit(int(current), int(total), int(frames), str(phase))
            if original is not None:
                original(current, total, frames, phase)
            _capture_preview(ctx, int(current), int(total), str(phase), overlap)

        kwargs["progress_callback"] = callback
        result = function(*args, **kwargs)
        _state.cursor.advance(
            _video_length(ctx),
            overlap,
            prepend_already_removed=requested_prepend > 0,
        )
        return result

    wrapped._videox_wrapped = True
    return wrapped


def _install_wrappers() -> None:
    import src.core.generation_phases as phases

    for name in _PHASE_FUNCTIONS:
        current = getattr(phases, name)
        wrapper = _wrap_postprocess if name == "postprocess_all_batches" else _wrap
        setattr(phases, name, wrapper(current))


def _bootstrap_seedvr() -> Path:
    root = Path(os.environ["VIDEOX_SEEDVR_ROOT"]).resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _configure_parent(argv: list[str]) -> None:
    input_path, skip, prepend = parse_preview_args(argv)
    _state.configure(input_path, skip, prepend, skip)


def _configure_worker(proc_idx: int, shared_args: dict, video_info: dict | None) -> None:
    info = video_info or {}
    source = info.get("video_path") or shared_args.get("input")
    skip = _safe_int(shared_args.get("skip_first_frames"), 0)
    prepend = _safe_int(shared_args.get("prepend_frames"), 0)
    if proc_idx != 0:
        prepend = 0
    start = _safe_int(info.get("start_frame"), skip)
    _state.configure(None if source is None else str(source), start, prepend, skip)


def _wrapped_worker(
    proc_idx,
    device_id,
    frames_np,
    shared_args,
    return_queue,
    done_barrier,
    video_info=None,
):
    _bootstrap_seedvr()
    _install_wrappers()
    _configure_worker(proc_idx, shared_args, video_info)
    import inference_cli

    return inference_cli._worker_process(
        proc_idx,
        device_id,
        frames_np,
        shared_args,
        return_queue,
        done_barrier,
        video_info=video_info,
    )


_wrapped_worker._videox_worker = True


def _install_worker_entry() -> None:
    import inference_cli

    if getattr(inference_cli._worker_process, "_videox_worker", False):
        return
    inference_cli._worker_process = _wrapped_worker


def _capture_preview(ctx, current: int, total: int, phase: str, overlap: int) -> None:
    directory = os.environ.get("VIDEOX_PREVIEW_DIR")
    if not directory or "post" not in phase.lower():
        return
    try:
        if not isinstance(ctx, dict):
            return
        info = ctx.get("decode_batch_info")
        if not isinstance(info, list) or not info:
            return
        batch_index = postprocess_batch_index(current, total, len(info))
        if batch_index is None:
            return
        write_end = int(info[batch_index][1])
        located = _state.cursor.locate(write_end - 1, overlap)
        if located is None:
            return
        source_index, frame_number = located
        output = _output_bgr(ctx.get("final_video"), write_end - 1)
        if output is None:
            return
        try:
            source = _read_source(source_index)
        except Exception:
            source = None
        _publish_pair(Path(directory), output, source, frame_number)
    except Exception:
        return


def _publish_pair(directory: Path, output, source, frame: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "output.jpg"
    source_path = directory / "source.jpg"
    _write_jpeg(output_path, output)
    if source is None:
        source_path.unlink(missing_ok=True)
    else:
        try:
            _write_jpeg(source_path, source)
        except Exception:
            source_path.unlink(missing_ok=True)
    _emit_preview(frame)


def _output_bgr(video, index: int):
    if video is None or not hasattr(video, "shape") or len(getattr(video, "shape", ())) < 3:
        return None
    if index < 0 or index >= int(video.shape[0]):
        return None
    frame = video[index].detach()
    if float(frame.min()) < -0.01:
        return None
    array = frame.float().clamp(0, 1).mul(255).round().cpu().numpy()
    if array.ndim != 3:
        return None
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        return None
    import numpy as np

    rgb = array.astype("uint8")
    return np.ascontiguousarray(rgb[..., ::-1])


def _read_source(index: int):
    path = _state.input_path
    if not path:
        return None
    import cv2

    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return None if image is None else _as_bgr(image)
    if _state._capture is None or not _state._capture.isOpened():
        _state.close()
        _state._capture = cv2.VideoCapture(path)
    capture = _state._capture
    if capture is None or not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return _as_bgr(frame)


def _as_bgr(image):
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[-1]
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _write_jpeg(path: Path, image) -> None:
    import cv2

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if not cv2.imwrite(str(temporary), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        temporary.unlink(missing_ok=True)
        raise OSError(f"无法写出预览: {path.name}")
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _video_length(ctx) -> int:
    if not isinstance(ctx, dict):
        return 0
    video = ctx.get("final_video")
    shape = getattr(video, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


def _argument(args, kwargs, name: str, index: int, default):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    root = _bootstrap_seedvr()
    saved_pythonpath = os.environ.get("PYTHONPATH", "")
    _configure_parent(sys.argv)
    _install_wrappers()
    import inference_cli

    parts = [str(root)]
    if saved_pythonpath:
        parts.append(saved_pythonpath)
    # The upstream CLI joins PYTHONPATH with ':' which Windows does not split.
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    _install_worker_entry()
    inference_cli.main()
    _state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
