"""Run SeedVR2 and emit one JSON progress object per line.

SeedVR2's phase functions accept a progress callback, but the CLI always
passes None. This entry wraps those functions before importing the CLI.
During post-processing it also writes the latest finished frame and the
matching source frame for the desktop preview.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
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
    Neither of those regions is part of the frames that get saved: multi-GPU
    runs drop the prepended frames after the workers finish, and single-GPU
    runs have postprocess_all_batches drop them (see _wrap_postprocess).
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
        # Upstream only removes prepended frames in its multi-GPU path. A
        # single-device parent asks postprocess_all_batches to drop them.
        self.strip_prepend = False
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
# Which worker this process is, how many share the job, and which streaming
# chunk is running, so the log parser can turn phase steps into one bar.
_job = {"worker": 0, "workers": 1, "chunk": 0, "chunks": 1}


def _emit(current: int, total: int, frames: int, phase: str) -> None:
    print(
        json.dumps(
            {
                "videox_progress": {
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "frames": frames,
                    "worker": _job["worker"],
                    "workers": _job["workers"],
                    "chunk": max(1, _job["chunk"]),
                    "chunks": max(1, _job["chunks"]),
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
        cursor = _state.cursor
        if _state.strip_prepend and cursor.calls == 0 and cursor.prepend > 0:
            kwargs["prepend_frames"] = cursor.prepend
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


def _install_chunk_tracking(cli) -> None:
    """Count streaming chunks; both functions are looked up by global name."""
    stream = cli._stream_video_chunks
    if getattr(stream, "_videox_wrapped", False):
        return
    core = cli._process_frames_core

    def counted_core(*args, **kwargs):
        _job["chunk"] += 1
        return core(*args, **kwargs)

    def counted_stream(*args, **kwargs):
        _job["chunks"] = max(1, _safe_int(kwargs.get("total_chunks"), 1))
        _job["chunk"] = 0
        yield from stream(*args, **kwargs)

    counted_stream._videox_wrapped = True
    cli._process_frames_core = counted_core
    cli._stream_video_chunks = counted_stream


def _install_flat_png_output(cli) -> None:
    """Write a video's PNG sequence into the folder VideoX passed, not <folder>/<stem>."""
    upstream = cli.generate_output_path
    if getattr(upstream, "_videox_wrapped", False):
        return

    def output_path(input_path, output_format, output_dir=None, input_type=None, from_directory=False):
        if output_dir and output_format == "png" and input_type == "video" and not from_directory:
            return str(Path(output_dir).resolve())
        return upstream(
            input_path,
            output_format,
            output_dir=output_dir,
            input_type=input_type,
            from_directory=from_directory,
        )

    output_path._videox_wrapped = True
    cli.generate_output_path = output_path


def _install_unicode_imageio() -> None:
    """cv2.imread/imwrite open paths with the ANSI code page on Windows.

    Route them through numpy file I/O so image jobs and PNG sequences work
    with Chinese (or any non-ASCII) paths.
    """
    import cv2
    import numpy as np

    if getattr(cv2.imread, "_videox_wrapped", False):
        return

    def imread(path, flags=cv2.IMREAD_COLOR):
        try:
            data = np.fromfile(os.fspath(path), dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(data, flags) if data.size else None

    def imwrite(path, image, params=()):
        path = os.fspath(path)
        ok, buffer = cv2.imencode(os.path.splitext(path)[1] or ".png", image, list(params))
        if not ok:
            return False
        try:
            buffer.tofile(path)
        except OSError:
            return False
        return True

    imread._videox_wrapped = imwrite._videox_wrapped = True
    cv2.imread, cv2.imwrite = imread, imwrite


_workers: list = []


def _install_worker_watchdog() -> None:
    """Fail the job when a multi-GPU worker dies.

    Upstream's parent waits for every worker's result with no timeout, so a
    worker that runs out of memory would leave the job running forever.
    """
    import multiprocessing.process as mp_process

    start = mp_process.BaseProcess.start
    if getattr(start, "_videox_wrapped", False):
        return

    def tracked_start(self):
        # BaseProcess.start deletes _target once the child is running.
        target = getattr(self, "_target", None)
        start(self)
        if target is _wrapped_worker:
            _workers.append(self)

    tracked_start._videox_wrapped = True
    mp_process.BaseProcess.start = tracked_start

    def watch() -> None:
        while True:
            time.sleep(2)
            dead = [worker for worker in list(_workers) if worker.exitcode not in (None, 0)]
            if not dead:
                continue
            code = dead[0].exitcode
            # Exit code 1 is an uncaught exception whose traceback the job error already
            # quotes; a killed or crashed worker leaves nothing, so this line is the error.
            prefix = "" if code == 1 else "错误: "
            cause = "系统内存或显存不足" if code == 1 else "系统内存不足（进程被强制结束）或显存不足"
            try:
                print(
                    f"{prefix}SeedVR2 多卡子进程异常退出（退出码 {code}），任务已终止。"
                    f"常见原因是{cause}，可以只选一张卡，或降低分辨率、批大小后重试。",
                    flush=True,
                )
            except (OSError, ValueError):
                # Nobody is reading any more (Ctrl-C); still stop the workers and exit.
                pass
            for worker in list(_workers):
                if worker.exitcode is None:
                    worker.terminate()
            os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _bootstrap_seedvr() -> Path:
    root = Path(os.environ["VIDEOX_SEEDVR_ROOT"]).resolve()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _configure_parent(argv: list[str]) -> None:
    input_path, skip, prepend = parse_preview_args(argv)
    _state.configure(input_path, skip, prepend, skip)
    devices = device_count(argv)
    _state.strip_prepend = devices <= 1
    _job["workers"] = devices


def device_count(argv: list[str]) -> int:
    """Number of GPUs in --cuda_device, split the way upstream main() splits it."""
    for index, token in enumerate(argv[:-1]):
        if token == "--cuda_device":
            return max(1, len([item for item in argv[index + 1].split(",") if item.strip()]))
    return 1


def _configure_worker(proc_idx: int, shared_args: dict, video_info: dict | None) -> None:
    info = video_info or {}
    source = info.get("video_path") or shared_args.get("input")
    skip = _safe_int(shared_args.get("skip_first_frames"), 0)
    prepend = _safe_int(shared_args.get("prepend_frames"), 0)
    if proc_idx != 0:
        prepend = 0
    start = _safe_int(info.get("start_frame"), skip)
    _state.configure(None if source is None else str(source), start, prepend, skip)
    _state.strip_prepend = False
    devices = str(shared_args.get("cuda_device") or "0")
    _job["worker"] = _safe_int(proc_idx, 0)
    _job["workers"] = max(1, len([item for item in devices.split(",") if item.strip()]))


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

    _install_chunk_tracking(inference_cli)
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

    # cv2 picks the encoder from the extension, so the temporary name keeps it.
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
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
    # Upstream sets this before torch is imported; _install_wrappers imports torch
    # first, so set it here or the single-GPU run keeps the default allocator.
    if platform.system() != "Darwin":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    _install_wrappers()
    import inference_cli

    parts = [str(root)]
    if saved_pythonpath:
        parts.append(saved_pythonpath)
    # The upstream CLI joins PYTHONPATH with ':' which Windows does not split.
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    _install_worker_entry()
    _install_worker_watchdog()
    _install_chunk_tracking(inference_cli)
    _install_flat_png_output(inference_cli)
    _install_unicode_imageio()
    inference_cli.main()
    _state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
