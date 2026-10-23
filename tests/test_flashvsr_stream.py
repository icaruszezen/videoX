"""The FlashVSR streaming loop, driven with stand-ins for the upstream models."""

from __future__ import annotations

import threading

import pytest

from videox.engines.flashvsr.stream import InputFrames, restore_stream, step_plan
from videox.errors import VideoXError

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

WIDTH, HEIGHT = 32, 16  # ×4 = 128x64, already multiples of 128 on the long side only


class FakeReader:
    """Flat grey frames whose level is the frame index, so a pixel tells which frame it came from."""

    def __init__(self, total: int, *, fail: bool = False) -> None:
        self.width, self.height = WIDTH, HEIGHT
        self.fps, self.rate, self.sar, self.frame_count = 25.0, "25/1", None, total
        self._total = total
        self._sent = 0
        self._fail = fail
        self.killed = False
        self.closed = False

    def read_many(self, count: int) -> list[bytes]:
        batch = []
        while len(batch) < count and self._sent < self._total:
            batch.append(bytes([self._sent % 256]) * (WIDTH * HEIGHT * 3))
            self._sent += 1
        return batch

    def finish(self) -> None:
        if self._fail:
            raise VideoXError("读取视频失败 (ffmpeg 1): broken")

    def kill(self) -> None:
        self.killed = True

    def close(self) -> None:
        self.closed = True


def frame_index(tensor) -> int:
    """Recover the source frame index from a flat frame in [-1, 1]."""
    return round((float(tensor.float().mean()) + 1) * 127.5)


def test_step_plan_matches_upstream_slicing():
    pieces, latents, end = step_plan(0)
    assert pieces == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 21), (21, 25)]
    assert (latents, end) == (6, 21)
    assert step_plan(1) == ([(25, 29), (29, 33)], 2, 29)
    assert step_plan(5) == ([(57, 61), (61, 65)], 2, 61)


def test_input_frames_pad_with_the_last_frame_and_drop_old_ones():
    frames = InputFrames(FakeReader(10), 4)
    try:
        assert (frames.padded_w, frames.padded_h) == (128, 128)
        clip = frames.clip(0, 25)
        assert tuple(clip.shape) == (1, 3, 25, 128, 128)
        assert [frame_index(clip[0, :, i]) for i in range(25)] == list(range(10)) + [9] * 15
        assert frames.total == 10
        frames.drop_before(21)
        # The last real frame stays: it stands in for everything past the end.
        assert frame_index(frames.clip(25, 26)[0, :, 0]) == 9
        assert frames.source(30) == bytes([9]) * (WIDTH * HEIGHT * 3)
    finally:
        frames.close()


def test_input_frames_reports_a_failed_decode_and_an_empty_clip():
    broken = InputFrames(FakeReader(3, fail=True), 4)
    try:
        with pytest.raises(VideoXError, match="broken"):
            broken.clip(0, 25)
    finally:
        broken.close()
    empty = InputFrames(FakeReader(0), 4)
    try:
        with pytest.raises(VideoXError, match="读不到视频帧"):
            empty.clip(0, 1)
    finally:
        empty.close()


def test_closing_early_stops_the_reader_thread():
    reader = FakeReader(10_000)
    frames = InputFrames(reader, 4, ahead=2)
    frames.clip(0, 1)
    frames.close()
    assert reader.closed
    assert not any(thread.name == "flashvsr-input" and thread.is_alive() for thread in threading.enumerate())


class FakeProjection:
    """Like Causal_LQ4x_Proj.stream_forward: the first clip only primes the cache."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.primed = False

    def clear_cache(self) -> None:
        self.primed = False

    def stream_forward(self, clip):
        self.calls.append([frame_index(clip[0, :, i]) for i in range(clip.shape[2])])
        if not self.primed:
            self.primed = True
            return None
        return [clip.float().mean().reshape(1, 1, 1).expand(1, clip.shape[2] // 2, 4)]


class FakeDecoder:
    """Returns the conditioning frames, so each output frame shows which input it belongs to."""

    def __init__(self) -> None:
        self.cleaned = 0

    def clean_mem(self) -> None:
        self.cleaned += 1

    def decode_video(self, latents, parallel, show_progress_bar, cond):
        assert parallel is False
        return (cond.transpose(1, 2).float() + 1) / 2


class FakeDit:
    def __init__(self) -> None:
        self.blocks = [object(), object()]
        self.LQ_proj_in = FakeProjection()
        self.freqs = (torch.zeros(1024, 2), torch.zeros(1024, 2), torch.zeros(1024, 2))


class FakePipe:
    def __init__(self) -> None:
        self.dit = FakeDit()
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.timestep = self.t = self.t_mod = None
        self.TCDecoder = FakeDecoder()
        self.steps: list[dict] = []

    def denoising_model(self):
        return self.dit

    def ColorCorrector(self, video, low, clip_range, chunk_size, method):  # noqa: N802 - upstream's name
        assert video.shape == low.shape
        return video


def model_fn_wan_video(dit, *, x, pre_cache_k, pre_cache_v, cur_process_idx, LQ_latents, **kwargs):
    """Stands in for upstream's function of the same name, looked up on the pipe's module."""
    assert kwargs["is_stream"] is True and kwargs["context"] is None and kwargs["kv_ratio"] == 3
    previous = pre_cache_k[0]
    dit.trace.append(
        {
            "step": cur_process_idx,
            "frames": x.shape[2],
            "previous": None if previous is None else int(previous[0]),
            "compact": all(
                cache is None or cache.untyped_storage().nbytes() == cache.nbytes for cache in pre_cache_k + pre_cache_v
            ),
            "lq": LQ_latents is not None,
        }
    )
    for caches in (pre_cache_k, pre_cache_v):
        for block in range(len(caches)):
            # Like upstream: the kept window is a slice of cat([history, current]).
            caches[block] = torch.full((2,), float(cur_process_idx))[1:]
    return torch.zeros_like(x), pre_cache_k, pre_cache_v


def run(total: int):
    pipe = FakePipe()
    pipe.dit.trace = []
    frames = InputFrames(FakeReader(total), 4)
    written: list[int] = []

    def write(start, video):
        assert start == len(written)
        count = video.shape[2] if frames.total is None else min(video.shape[2], frames.total - start)
        written.extend(frame_index(video[0, :, i]) for i in range(count))

    try:
        restore_stream(
            pipe,
            frames,
            write,
            full=False,
            seed=0,
            sparse_ratio=2.0,
            kv_ratio=3,
            local_range=11,
            color_fix=True,
        )
    finally:
        frames.close()
    return pipe, written


@pytest.mark.parametrize("total", [1, 20, 21, 22, 29, 30, 37, 100])
def test_every_source_frame_comes_out_once_and_in_order(total: int):
    pipe, written = run(total)
    assert written == list(range(total))
    # Upstream pads to the smallest 8n+1 >= max(total + 4, 25) and runs n - 2 steps.
    padded = ((max(total + 4, 25) - 1 + 7) // 8) * 8 + 1
    assert len(pipe.dit.trace) == (padded - 1) // 8 - 2


def test_steps_carry_the_cache_and_feed_upstreams_input_slices():
    pipe, _written = run(36)
    trace = pipe.dit.trace
    assert [step["frames"] for step in trace] == [6, 2, 2]
    assert [step["previous"] for step in trace] == [None, 0, 1]
    # The dropped history segment is freed between steps, not kept alive by a view.
    assert all(step["compact"] for step in trace)
    assert all(step["lq"] for step in trace)
    calls = pipe.dit.LQ_proj_in.calls
    assert calls[:7] == [[0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    # Past frame 35 the last frame repeats, as in upstream's padded clip.
    assert calls[7:] == [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 35], [35, 35, 35, 35]]
    assert pipe.TCDecoder.cleaned == 1
