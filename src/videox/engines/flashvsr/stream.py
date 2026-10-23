"""Restore a clip 8 frames at a time with FlashVSR v1.1's streaming loop.

Upstream's pipelines take the whole clip in one call: the enlarged input, the
noise for every latent frame (allocated on the GPU) and every decoded frame stay
in memory until the end, so an hour of video cannot fit. restore_stream runs the
same per-step loop as FlashVSRTinyLongPipeline.__call__, but reads the input,
draws the noise and hands each decoded step to the writer as it goes. Only a few
dozen frames and the model's own bounded caches are held at any time.
"""

from __future__ import annotations

import importlib
import os
import queue
import threading
import time
from pathlib import Path

from videox.engines.preview_io import publish_pair
from videox.engines.progress import emit_progress
from videox.errors import VideoXError
from videox.media.frames import BT709_OUTPUT_FILTER

# Seconds between preview updates; each one runs ffmpeg twice.
PREVIEW_INTERVAL = 10.0
_END = object()


def step_plan(step: int) -> tuple[list[tuple[int, int]], int, int]:
    """What one step of the loop consumes and produces.

    Returns the input frame slices fed to the LQ projection, the number of
    latent frames denoised, and the index one past the last frame it outputs.
    The first step outputs 21 frames, every later step 8; each step reads 4
    frames ahead of what it outputs.
    """
    if step == 0:
        return [(max(0, index * 4 - 3), (index + 1) * 4 - 3) for index in range(7)], 6, 21
    base = step * 8 + 17
    return [(base, base + 4), (base + 4, base + 8)], 2, step * 8 + 21


def draw_noise(generator, count: int, height: int, width: int):
    import torch

    return torch.randn((1, 16, count, height // 8, width // 8), generator=generator, dtype=torch.float32)


def ensure_rope_length(dit, positions: int) -> None:
    """Grow the DiT's temporal RoPE table to cover `positions` latent frames.

    Upstream builds 1024 positions, about 4100 frames of video, and step k uses
    positions up to 6 + 2k. Longer tables share their first rows with the
    default one, and the cached keys carry their own positions, so attention
    only sees the distance between nearby steps.
    """
    have = int(dit.freqs[0].shape[0])
    if have >= positions:
        return
    from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d

    head_dim = 2 * sum(int(table.shape[1]) for table in dit.freqs)
    dit.freqs = precompute_freqs_cis_3d(head_dim, end=max(positions, 2 * have))


def restore_stream(
    pipe,
    frames,
    write,
    *,
    full: bool,
    seed: int,
    sparse_ratio: float,
    kv_ratio: int,
    local_range: int,
    color_fix: bool,
) -> None:
    """Run the streaming loop until every source frame has been written.

    `frames` provides clip(start, stop), drop_before(index), total (None until
    the end of the input is known) and padded_h/padded_w; `write(start, video)`
    receives each step's frames as a (1, 3, n, H, W) tensor in [-1, 1].
    """
    import torch

    model_fn = importlib.import_module(type(pipe).__module__).model_fn_wan_video
    dit = pipe.denoising_model()
    device = pipe.device
    height, width = frames.padded_h, frames.padded_w
    topk_ratio = sparse_ratio * 768 * 1280 / float(height * width)
    generator = torch.Generator("cpu").manual_seed(seed)
    dit.LQ_proj_in.clear_cache()
    if full:
        pipe.vae.clear_cache()
    else:
        pipe.TCDecoder.clean_mem()
    cache_k = [None] * len(dit.blocks)
    cache_v = [None] * len(dit.blocks)
    done = 0
    step = 0
    with torch.no_grad():
        while frames.total is None or done < frames.total:
            pieces, latent_count, end = step_plan(step)
            lq_latents = None
            for start, stop in pieces:
                projected = dit.LQ_proj_in.stream_forward(frames.clip(start, stop).to(device))
                if projected is None:
                    continue
                if lq_latents is None:
                    lq_latents = projected
                else:
                    lq_latents = [torch.cat(pair, dim=1) for pair in zip(lq_latents, projected)]
            ensure_rope_length(dit, 6 + 2 * step)
            noise = draw_noise(generator, latent_count, height, width).to(device=device, dtype=pipe.torch_dtype)
            noise_pred, cache_k, cache_v = model_fn(
                dit,
                x=noise,
                timestep=pipe.timestep,
                context=None,
                tea_cache=None,
                use_unified_sequence_parallel=False,
                LQ_latents=lq_latents,
                is_full_block=False,
                is_stream=True,
                pre_cache_k=cache_k,
                pre_cache_v=cache_v,
                topk_ratio=topk_ratio,
                kv_ratio=kv_ratio,
                cur_process_idx=step,
                t_mod=pipe.t_mod,
                t=pipe.t,
                local_range=local_range,
            )
            compact(cache_k)
            compact(cache_v)
            latents = noise - noise_pred
            del noise, noise_pred, lq_latents, projected
            low = frames.clip(done, end).to(device)
            if full:
                # The Wan VAE decoder is causal; its feature cache carries over between steps.
                video = pipe.vae.model.stream_decode(latents, pipe.vae.scale).clamp_(-1, 1)
            else:
                video = (
                    pipe.TCDecoder.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=low)
                    .transpose(1, 2)
                    .mul_(2)
                    .sub_(1)
                )
            if color_fix:
                video = pipe.ColorCorrector(video, low, clip_range=(-1, 1), chunk_size=None, method="adain")
            if int(video.shape[2]) != end - done:
                raise VideoXError(f"FlashVSR 第 {step + 1} 段输出了 {int(video.shape[2])} 帧，应为 {end - done} 帧")
            write(done, video)
            del latents, low, video
            frames.drop_before(end)
            done = end
            step += 1


def compact(caches: list) -> None:
    """Copy each block's history out of the buffer it was sliced from.

    Upstream returns the kept window as a view of cat([history, current]), so
    the oldest step stays allocated until the next step replaces it. Copying
    frees it before the decoder runs, which is the costliest part of a step;
    the values, and so the output, are unchanged.
    """
    for index, cache in enumerate(caches):
        if cache is not None and cache.untyped_storage().nbytes() > cache.nbytes:
            caches[index] = cache.clone()


def target_size(width: int, height: int, scale: int) -> tuple[int, int, int, int]:
    """The ×N size, and that size padded up to the multiple of 128 FlashVSR needs."""
    scaled_w = int(round(width * scale))
    scaled_h = int(round(height * scale))
    if scaled_w < 1 or scaled_h < 1:
        raise VideoXError(f"读不到有效的画面尺寸: {width}x{height}")
    padded_w = -(-scaled_w // 128) * 128
    padded_h = -(-scaled_h // 128) * 128
    return scaled_w, scaled_h, padded_w, padded_h


class InputFrames:
    """Source frames, enlarged and edge-padded for the model, decoded ahead on a thread.

    Holds only the frames the loop can still ask for plus a short read-ahead.
    Past the end of the clip the last frame stands in, the way upstream pads a
    clip to 8n+1 frames.
    """

    def __init__(self, reader, scale: int, ahead: int = 16) -> None:
        self.width, self.height = reader.width, reader.height
        self.scaled_w, self.scaled_h, self.padded_w, self.padded_h = target_size(reader.width, reader.height, scale)
        self.total: int | None = None
        self.reader = reader
        self._frames: dict[int, tuple[bytes, object]] = {}
        self._next = 0
        self._queue: queue.Queue = queue.Queue(maxsize=ahead)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._decode, name="flashvsr-input", daemon=True)
        self._thread.start()

    def clip(self, start: int, stop: int):
        """Frames start..stop-1 as a (1, 3, n, H, W) bfloat16 tensor on the CPU."""
        import torch

        self._fill(stop)
        last = self._next - 1
        try:
            tensors = [self._frames[min(index, last)][1] for index in range(start, stop)]
        except KeyError as exc:
            raise VideoXError(f"FlashVSR 要读的第 {exc.args[0] + 1} 帧已经释放") from exc
        return torch.stack(tensors, dim=1).unsqueeze(0)

    def source(self, index: int) -> bytes:
        """The decoded RGB bytes of source frame `index`, for the preview."""
        return self._frames[min(index, self._next - 1)][0]

    def drop_before(self, index: int) -> None:
        # Keep the newest frame: it stands in for the frames past the end.
        keep_from = min(index, self._next - 1)
        for key in [key for key in self._frames if key < keep_from]:
            del self._frames[key]

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            # Unblocks a read that is waiting on ffmpeg.
            self.reader.kill()
        self._thread.join(timeout=30)
        self.reader.close()

    def _fill(self, stop: int) -> None:
        while self.total is None and self._next < stop:
            item = self._queue.get()
            if item is _END:
                if self._next == 0:
                    raise VideoXError("读不到视频帧")
                self.total = self._next
            elif isinstance(item, BaseException):
                raise item
            else:
                self._frames[self._next] = item
                self._next += 1

    def _decode(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self.reader.read_many(8)
                if not batch:
                    self.reader.finish()
                    self._put(_END)
                    return
                for blob in batch:
                    if not self._put((blob, self._prepare(blob))):
                        return
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            self._put(exc)

    def _prepare(self, blob: bytes):
        from PIL import Image

        image = Image.frombytes("RGB", (self.width, self.height), blob)
        return to_tensor(upscale_pad(image, self.scaled_w, self.scaled_h, self.padded_w, self.padded_h))

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False


def upscale_pad(image, scaled_w: int, scaled_h: int, padded_w: int, padded_h: int):
    import numpy as np
    from PIL import Image

    enlarged = np.asarray(image.resize((scaled_w, scaled_h), Image.BICUBIC), dtype=np.uint8)
    return np.pad(
        enlarged,
        ((0, padded_h - scaled_h), (0, padded_w - scaled_w), (0, 0)),
        mode="edge",
    )


def to_tensor(image):
    import numpy as np
    import torch

    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255).mul(2).sub(1)
    return tensor.to(dtype=torch.bfloat16)


class OutputVideo:
    """Writes each step's frames as they arrive, cropped back to the plain ×N size."""

    def __init__(self, path: Path, frames: InputFrames, fps: float, rate: str | None, sar: str | None, estimate: int):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        # imageio writes -r with two decimals; a later input -r keeps 30000/1001 exact.
        extra = {"input_params": ["-r", rate]} if rate else {}
        filters = BT709_OUTPUT_FILTER + (f",setsar={sar}" if sar else "")
        self._writer = imageio.get_writer(
            str(path), fps=fps, quality=8, macro_block_size=None, output_params=["-vf", filters], **extra
        )
        self._frames = frames
        self._estimate = max(int(estimate), 1)
        self._preview_at: float | None = None
        self.written = 0

    def write(self, start: int, video) -> None:
        import torch

        if start != self.written:
            raise VideoXError(f"FlashVSR 输出不连续：已写 {self.written} 帧，又收到从第 {start + 1} 帧开始的画面")
        count = int(video.shape[2])
        if self._frames.total is not None:
            # The last step runs past the end on copies of the last frame.
            count = min(count, self._frames.total - start)
        height, width = self._frames.scaled_h, self._frames.scaled_w
        last = None
        for index in range(count):
            # One frame at a time keeps the float copy small at 4K.
            frame = video[0, :, index, :height, :width]
            pixels = ((frame.float() + 1) * 127.5).clamp_(0, 255).to(torch.uint8)
            last = pixels.permute(1, 2, 0).contiguous().cpu().numpy()
            self._writer.append_data(last)
        self.written = start + max(count, 0)
        # The probed frame count is an estimate; never let the bar reach the end early.
        emit_progress("restore", self.written, max(self._estimate, self.written + 1))
        now = time.monotonic()
        if last is not None and (self._preview_at is None or now - self._preview_at >= PREVIEW_INTERVAL):
            self._preview_at = now
            publish_pair(
                os.environ.get("VIDEOX_PREVIEW_DIR"),
                self.written,
                last.tobytes(),
                width,
                height,
                self._frames.source(self.written - 1),
                self._frames.width,
                self._frames.height,
            )

    def close(self) -> None:
        self._writer.close()
