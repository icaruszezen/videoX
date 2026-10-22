"""FlashVSR v1.1 inference. Imports the upstream checkout, which lives in its own venv."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from videox.engines.flashvsr.presets import weights_for
from videox.engines.flashvsr.weights import ensure_weights
from videox.engines.preview_io import publish_pair
from videox.engines.progress import emit_progress
from videox.errors import VideoXError
from videox.media.frames import probe_video


@dataclass(frozen=True)
class FlashRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    mode: str
    scale: int
    sparse_ratio: float
    local_range: int
    kv_ratio: int
    seed: int
    color_fix: bool


def run_flashvsr(request: FlashRequest) -> None:
    if request.mode not in {"tiny-long", "tiny", "full"}:
        raise VideoXError(f"模式无法识别: {request.mode}")
    if not request.input_path.is_file():
        raise VideoXError(f"输入文件不存在: {request.input_path}")
    if not _sparse_kernel_ready():
        # diffsynth imports block_sparse_attn unconditionally, so there is no
        # full-attention fallback to switch to.
        raise VideoXError(
            "FlashVSR 需要 Block-Sparse-Attention（稀疏注意力内核），当前环境没有装好。"
            "请安装 CUDA Toolkit 后重新运行 videox setup。"
        )
    _patch_tqdm()
    # Fail on an unreadable input before downloading and loading several GB of models.
    probe_video(request.input_path)
    emit_progress("download", 0, 1)
    ensure_weights(request.model_dir, weights_for(request.mode))
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    pipe = _init_pipeline(request.model_dir, request.mode)
    emit_progress("load", 1, 1)
    emit_progress("restore", 0, 1)
    clip = _prepare_input(request.input_path, request.scale)
    low_quality = clip.video
    if request.mode != "tiny-long":
        # Only the Tiny Long pipeline moves each slice of the input to the GPU itself.
        low_quality = low_quality.to(pipe.device)
    height, width = clip.padded_h, clip.padded_w
    _ensure_rope_length(pipe, clip.count)
    topk = request.sparse_ratio * 768 * 1280 / float(height * width)
    kwargs = {
        "prompt": "",
        "negative_prompt": "",
        "cfg_scale": 1.0,
        "num_inference_steps": 1,
        "seed": request.seed,
        "LQ_video": low_quality,
        "num_frames": clip.count,
        "height": height,
        "width": width,
        "is_full_block": False,
        "if_buffer": True,
        "topk_ratio": topk,
        "kv_ratio": request.kv_ratio,
        "local_range": request.local_range,
        "color_fix": request.color_fix,
    }
    if request.mode == "full":
        kwargs["tiled"] = False
    video = pipe(**kwargs)
    del low_quality
    kwargs.clear()
    clip.video = None
    # The input was padded with copies of the last frame and edge pixels; keep
    # only the source frames at the plain ×N size.
    video = video[:, : clip.total, : clip.scaled_h, : clip.scaled_w]
    _save_video(video, request.output_path, clip.fps, clip.rate)
    _publish(video, clip.source)
    emit_progress("restore", 1, 1)


def _ensure_rope_length(pipe, frames: int) -> None:
    """Grow the DiT's temporal RoPE table for clips longer than about 4100 frames.

    Upstream builds 1024 positions and indexes them at 4 + 2 * step. Longer
    tables share their first 1024 rows with the default one.
    """
    from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d

    dit = pipe.denoising_model()
    needed = 4 + 2 * ((frames - 1) // 8) + 8
    if dit.freqs[0].shape[0] >= needed:
        return
    head_dim = 2 * sum(int(table.shape[1]) for table in dit.freqs)
    dit.freqs = precompute_freqs_cis_3d(head_dim, end=needed)


def _sparse_kernel_ready() -> bool:
    try:
        import torch

        torch.cuda.is_available()
        import block_sparse_attn  # noqa: F401
    except Exception:
        return False
    return True


def _patch_tqdm() -> None:
    import tqdm as tqdm_module

    original = tqdm_module.tqdm

    class ReportingTqdm(original):
        def update(self, n=1):
            result = super().update(n)
            total = self.total or 0
            if total:
                emit_progress("restore", float(self.n or 0), float(total))
            return result

    tqdm_module.tqdm = ReportingTqdm
    try:
        import tqdm.auto as auto

        auto.tqdm = ReportingTqdm
    except Exception:
        return


def _init_pipeline(model_dir: Path, mode: str):
    import torch
    from diffsynth import ModelManager
    from utils.utils import Causal_LQ4x_Proj

    dit = model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"
    projection = model_dir / "LQ_proj_in.ckpt"
    manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    if mode == "full":
        from diffsynth import FlashVSRFullPipeline

        manager.load_models([str(dit), str(model_dir / "Wan2.1_VAE.pth")])
        pipe = FlashVSRFullPipeline.from_model_manager(manager, device="cuda")
    elif mode == "tiny":
        from diffsynth import FlashVSRTinyPipeline

        manager.load_models([str(dit)])
        pipe = FlashVSRTinyPipeline.from_model_manager(manager, device="cuda")
    else:
        from diffsynth import FlashVSRTinyLongPipeline

        manager.load_models([str(dit)])
        pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        "cuda", dtype=torch.bfloat16
    )
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(projection, map_location="cpu", weights_only=False),
        strict=True,
    )
    pipe.denoising_model().LQ_proj_in.to("cuda")
    if mode == "full":
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        from utils.TCDecoder import build_tcdecoder

        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
        pipe.TCDecoder.load_state_dict(
            torch.load(model_dir / "TCDecoder.ckpt", map_location="cpu", weights_only=False),
            strict=False,
        )
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


@dataclass
class _Clip:
    video: object
    total: int
    count: int
    scaled_w: int
    scaled_h: int
    padded_w: int
    padded_h: int
    fps: float
    rate: str
    source: object


def _prepare_input(path: Path, scale: int) -> _Clip:
    import numpy as np
    import torch
    from PIL import Image

    from videox.media.frames import FrameReader

    # ffmpeg, not imageio: imageio cannot open .m4v/.flv/.3gp/.ts by itself and
    # VFR clips need the same constant rate the output is written at.
    reader = FrameReader(path)
    try:
        frames: list[bytes] = []
        while batch := reader.read_many(64):
            frames.extend(batch)
        reader.finish()
    finally:
        reader.close()
    total = len(frames)
    if total <= 0:
        raise VideoXError(f"读不到视频帧: {path}")
    width, height = reader.width, reader.height
    scaled_w, scaled_h, padded_w, padded_h = _target_size(width, height, scale)
    print(f"原片 {width}x{height}，{total} 帧，{reader.fps:g} fps；输出 {scaled_w}x{scaled_h}", flush=True)
    source = np.frombuffer(frames[0], dtype=np.uint8).reshape(height, width, 3).copy()
    # The pipeline needs 8k+1 frames and drops the last 4; pad with the last
    # frame instead of cutting the tail off, and trim the result later.
    count = _smallest_8n1_at_least(max(total + 4, 25))
    video = torch.empty((1, 3, count, padded_h, padded_w), dtype=torch.bfloat16)
    for slot in range(total):
        image = Image.frombytes("RGB", (width, height), frames[slot])
        frames[slot] = b""
        video[0, :, slot] = _to_tensor(_upscale_pad(image, scaled_w, scaled_h, padded_w, padded_h))
    video[0, :, total:] = video[0, :, total - 1 : total]
    return _Clip(video, total, count, scaled_w, scaled_h, padded_w, padded_h, reader.fps, reader.rate, source)


def _smallest_8n1_at_least(count: int) -> int:
    return ((max(count, 1) - 1 + 7) // 8) * 8 + 1


def _target_size(width: int, height: int, scale: int) -> tuple[int, int, int, int]:
    """The ×N size, and that size padded up to the multiple of 128 FlashVSR needs."""
    scaled_w = int(round(width * scale))
    scaled_h = int(round(height * scale))
    if scaled_w < 1 or scaled_h < 1:
        raise VideoXError(f"读不到有效的画面尺寸: {width}x{height}")
    padded_w = -(-scaled_w // 128) * 128
    padded_h = -(-scaled_h // 128) * 128
    return scaled_w, scaled_h, padded_w, padded_h


def _upscale_pad(image, scaled_w: int, scaled_h: int, padded_w: int, padded_h: int):
    import numpy as np
    from PIL import Image

    enlarged = np.asarray(image.resize((scaled_w, scaled_h), Image.BICUBIC), dtype=np.uint8)
    return np.pad(
        enlarged,
        ((0, padded_h - scaled_h), (0, padded_w - scaled_w), (0, 0)),
        mode="edge",
    )


def _to_tensor(image):
    import numpy as np
    import torch

    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255).mul(2).sub(1)
    return tensor.to(dtype=torch.bfloat16)


def _save_video(video, path: Path, fps: float, rate: str | None) -> None:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from einops import rearrange

    path.parent.mkdir(parents=True, exist_ok=True)
    # imageio writes -r with two decimals; a later input -r keeps 30000/1001 exact.
    extra = {"input_params": ["-r", rate]} if rate else {}
    writer = imageio.get_writer(str(path), fps=fps, quality=8, macro_block_size=None, **extra)
    try:
        # Convert a few frames at a time on the CPU instead of the whole clip as float32.
        for start in range(0, int(video.shape[1]), 8):
            chunk = rearrange(video[:, start : start + 8], "C T H W -> T H W C")
            chunk = ((chunk.detach().to("cpu", torch.float32) + 1) * 127.5).clamp(0, 255).byte().numpy()
            for frame in chunk:
                writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    if not path.is_file():
        raise VideoXError(f"引擎没有写出视频: {path}")


def _publish(video, source) -> None:
    import numpy as np

    frame = video[:, 0].detach().float()
    array = ((frame + 1) * 127.5).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    source_array = np.asarray(source, dtype=np.uint8)
    publish_pair(
        os.environ.get("VIDEOX_PREVIEW_DIR"),
        1,
        array.tobytes(),
        int(array.shape[1]),
        int(array.shape[0]),
        source_array.tobytes(),
        int(source_array.shape[1]),
        int(source_array.shape[0]),
    )
