"""FlashVSR v1.1 inference. Imports the upstream checkout, which lives in its own venv."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from videox.engines.flashvsr.weights import ensure_weights
from videox.engines.preview_io import publish_pair
from videox.engines.progress import emit_progress
from videox.errors import VideoXError


@dataclass(frozen=True)
class FlashRequest:
    input_path: Path
    output_path: Path
    model_dir: Path
    mode: str
    scale: int
    sparse_ratio: float
    local_range: int
    kv_ratio: float
    seed: int
    color_fix: bool


def run_flashvsr(request: FlashRequest) -> None:
    if request.mode not in {"tiny-long", "tiny", "full"}:
        raise VideoXError(f"模式无法识别: {request.mode}")
    if not request.input_path.is_file():
        raise VideoXError(f"输入文件不存在: {request.input_path}")
    _patch_tqdm()
    emit_progress("download", 0, 1)
    ensure_weights(request.model_dir)
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    full_block = not _sparse_kernel_ready()
    if full_block:
        print("没有 Block-Sparse-Attention，改用整块注意力。速度会慢一些。", flush=True)
    pipe = _init_pipeline(request.model_dir, request.mode)
    emit_progress("load", 1, 1)
    low_quality, height, width, frames, fps, source = _prepare_input(request.input_path, request.scale)
    topk = request.sparse_ratio * 768 * 1280 / float(height * width)
    kwargs = {
        "prompt": "",
        "negative_prompt": "",
        "cfg_scale": 1.0,
        "num_inference_steps": 1,
        "seed": request.seed,
        "LQ_video": low_quality,
        "num_frames": frames,
        "height": height,
        "width": width,
        "is_full_block": full_block,
        "if_buffer": True,
        "topk_ratio": topk,
        "kv_ratio": request.kv_ratio,
        "local_range": request.local_range,
        "color_fix": request.color_fix,
    }
    if request.mode == "full":
        kwargs["tiled"] = False
    video = pipe(**kwargs)
    _save_video(video, request.output_path, fps)
    _publish(video, source)
    emit_progress("restore", 1, 1)


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


def _prepare_input(path: Path, scale: int):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image

    reader = imageio.get_reader(str(path))
    try:
        first = Image.fromarray(reader.get_data(0)).convert("RGB")
        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            meta = {}
        fps_value = meta.get("fps", 30)
        fps = int(round(fps_value)) if isinstance(fps_value, (int, float)) else 30
        total = _count_frames(reader, meta)
        if total <= 0:
            raise VideoXError(f"读不到视频帧: {path}")
        width, height = first.size
        scaled_w, scaled_h, target_w, target_h = _target_size(width, height, scale)
        print(
            f"原片 {width}x{height}，{total} 帧，{fps} fps；输出 {target_w}x{target_h}",
            flush=True,
        )
        indexes = list(range(total)) + [total - 1] * 4
        count = _largest_8n1(len(indexes))
        if count == 0:
            raise VideoXError("视频太短，无法组成 FlashVSR 需要的帧数")
        indexes = indexes[:count]
        frames = []
        for index in indexes:
            image = Image.fromarray(reader.get_data(index)).convert("RGB")
            frames.append(_to_tensor(_center_crop(image, scale, target_w, target_h)))
    finally:
        reader.close()
    video = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
    source = np.asarray(first, dtype=np.uint8)
    return video, target_h, target_w, count, max(fps, 1), source


def _count_frames(reader, meta: dict) -> int:
    count = meta.get("nframes")
    if isinstance(count, int) and count > 0:
        return count
    try:
        counted = reader.count_frames()
        if isinstance(counted, int) and counted > 0:
            return counted
    except Exception:
        return 0
    return 0


def _largest_8n1(count: int) -> int:
    if count < 1:
        return 0
    return ((count - 1) // 8) * 8 + 1


def _target_size(width: int, height: int, scale: int) -> tuple[int, int, int, int]:
    scaled_w = int(round(width * scale))
    scaled_h = int(round(height * scale))
    target_w = (scaled_w // 128) * 128
    target_h = (scaled_h // 128) * 128
    if target_w < 128 or target_h < 128:
        raise VideoXError(
            f"放大后的画面 {scaled_w}x{scaled_h} 小于 128 像素。FlashVSR 需要至少一边能裁成 128 的倍数。"
        )
    return scaled_w, scaled_h, target_w, target_h


def _center_crop(image, scale: int, target_w: int, target_h: int):
    from PIL import Image

    width, height = image.size
    scaled_w = int(round(width * scale))
    scaled_h = int(round(height * scale))
    enlarged = image.resize((scaled_w, scaled_h), Image.BICUBIC)
    left = max(0, (scaled_w - target_w) // 2)
    top = max(0, (scaled_h - target_h) // 2)
    return enlarged.crop((left, top, left + target_w, top + target_h))


def _to_tensor(image):
    import numpy as np
    import torch

    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255).mul(2).sub(1)
    return tensor.to(dtype=torch.bfloat16)


def _save_video(video, path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from einops import rearrange

    frames = rearrange(video, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=8, macro_block_size=None)
    try:
        for frame in frames:
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
