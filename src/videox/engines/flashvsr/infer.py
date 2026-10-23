"""FlashVSR v1.1 inference. Imports the upstream checkout, which lives in its own venv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from videox.engines.flashvsr.presets import weights_for
from videox.engines.flashvsr.stream import InputFrames, OutputVideo, restore_stream, target_size
from videox.engines.flashvsr.weights import ensure_weights
from videox.engines.progress import emit_progress
from videox.errors import VideoXError
from videox.media.frames import FrameReader, probe_video


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
    import torch

    # Fail on an unreadable input, and show what the GPU must hold, before
    # downloading and loading several GB of models.
    probe = probe_video(request.input_path)
    scaled_w, scaled_h, padded_w, padded_h = target_size(probe.width, probe.height, request.scale)
    full = request.mode == "full"
    need = vram_estimate_gb(padded_w, padded_h, request.kv_ratio, full=full)
    history = (request.kv_ratio + 1) * padded_w * padded_h * HISTORY_BYTES / GB
    detail = f"其中历史参考（KV 缓存）约 {history:.1f} GB"
    if full:
        detail += f"，Full 模式的解码缓存约 {padded_w * padded_h * VAE_CACHE_BYTES / GB:.1f} GB"
    print(
        f"原片 {probe.width}x{probe.height}，约 {probe.frame_count} 帧，{probe.fps:g} fps；"
        f"输出 {scaled_w}x{scaled_h}，每 8 帧一段边读边处理，显存占用和视频长短无关；"
        f"按这个输出尺寸和 KV 比 {request.kv_ratio}，估计需要约 {need:.0f} GB 显存（{detail}）",
        flush=True,
    )
    card = _card_gb(torch)
    advice = _advice(request)
    if card is not None and need > card:
        print(f"注意：这张显卡只有 {card:.0f} GB，很可能显存不足。{advice}", flush=True)
    emit_progress("download", 0, 1)
    ensure_weights(request.model_dir, weights_for(request.mode))
    emit_progress("download", 1, 1)
    emit_progress("load", 0, 1)
    pipe = _init_pipeline(request.model_dir, request.mode)
    emit_progress("load", 1, 1)
    emit_progress("restore", 0, 1)
    frames = InputFrames(FrameReader(request.input_path), request.scale)
    try:
        reader = frames.reader
        output = OutputVideo(request.output_path, frames, reader.fps, reader.rate, reader.sar, reader.frame_count)
        try:
            restore_stream(
                pipe,
                frames,
                output.write,
                full=full,
                seed=request.seed,
                sparse_ratio=request.sparse_ratio,
                kv_ratio=request.kv_ratio,
                local_range=request.local_range,
                color_fix=request.color_fix,
            )
        except torch.cuda.OutOfMemoryError as exc:
            on_card = f"，这张显卡 {card:.0f} GB" if card is not None else ""
            raise VideoXError(
                f"显存不足：输出 {scaled_w}x{scaled_h}、KV 比 {request.kv_ratio} 时估计需要约 {need:.0f} GB 显存"
                f"{on_card}。FlashVSR 逐段处理，显存和视频长短无关，只随输出的宽×高和 KV 比增长。{advice}"
            ) from exc
        finally:
            output.close()
    finally:
        frames.close()
    if not request.output_path.is_file():
        raise VideoXError(f"引擎没有写出视频: {request.output_path}")
    emit_progress("restore", 1, 1)


GB = 1024**3
# GPU memory per output pixel, from replaying upstream's modules in bf16 at
# 1920x1152 output. The working-memory figures were measured on the CPU, so
# the estimate is rough.
MODEL_GB = 3.5  # weights and the CUDA context
HISTORY_BYTES = 1440  # one step of K and V: 2 latent frames, 30 blocks, 1536 values per token
DIT_BYTES = 630  # one DiT block's working memory
DECODER_BYTES = 2700  # TCDecoder's working memory while a step is decoded
VAE_BYTES = 5000  # the Wan VAE's working memory while a step is decoded
VAE_CACHE_BYTES = 4729  # Full: the Wan VAE's causal cache, kept between steps


def vram_estimate_gb(padded_w: int, padded_h: int, kv_ratio: int, *, full: bool) -> float:
    """Peak GPU memory of one step, in GiB; it does not depend on the length of the video.

    While the DiT runs it holds kv_ratio past steps plus the current one;
    restore_stream trims that back to kv_ratio before the decoder runs.
    """
    dit = (kv_ratio + 1) * HISTORY_BYTES + DIT_BYTES
    decode = kv_ratio * HISTORY_BYTES + (VAE_BYTES if full else DECODER_BYTES)
    cache = VAE_CACHE_BYTES if full else 0
    return MODEL_GB + padded_w * padded_h * (cache + max(dit, decode)) / GB


def _card_gb(torch) -> float | None:
    try:
        return torch.cuda.get_device_properties(0).total_memory / GB
    except Exception:
        return None


def _advice(request: FlashRequest) -> str:
    options = []
    if request.mode == "full":
        options.append("改用 Tiny Long 模式")
    if request.scale > 2:
        options.append("改用 2 倍放大")
    if request.kv_ratio > 2:
        options.append("把 KV 比调到 2")
    options.append("先把原片缩小")
    if len(options) == 1:
        return f"请{options[0]}。"
    return f"请{'、'.join(options[:-1])}，或{options[-1]}。"


def _sparse_kernel_ready() -> bool:
    try:
        import torch

        torch.cuda.is_available()
        import block_sparse_attn  # noqa: F401
    except Exception:
        return False
    return True


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
    else:
        # Tiny only differs from Tiny Long in keeping the whole clip on the GPU;
        # restore_stream drives both the same way.
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
