from __future__ import annotations

DEFAULT_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"
DEFAULT_RESOLUTION = 1080
DEFAULT_TEMPORAL_OVERLAP = 3
DEFAULT_PREPEND_FRAMES = 4
DEFAULT_COLOR_CORRECTION = "lab"
DEFAULT_SEED = 42
DEFAULT_ATTENTION_MODE = "sdpa"

REQUIRED_MODELS = (DEFAULT_DIT_MODEL, DEFAULT_VAE_MODEL)


def params_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "dit_model": {"type": "string", "default": DEFAULT_DIT_MODEL},
            "resolution": {"type": "integer", "default": DEFAULT_RESOLUTION, "minimum": 1},
            "batch_size": {
                "type": "integer",
                "description": "必须是 4n+1。省略时按显存选择：约 12GB 用 5，约 16GB 及以上用 9。",
            },
            "temporal_overlap": {"type": "integer", "default": DEFAULT_TEMPORAL_OVERLAP, "minimum": 0},
            "prepend_frames": {"type": "integer", "default": DEFAULT_PREPEND_FRAMES, "minimum": 0},
            "vae_encode_tiled": {"type": "boolean", "default": True},
            "vae_decode_tiled": {"type": "boolean", "default": True},
            "uniform_batch_size": {"type": "boolean", "default": True},
            "chunk_size": {"type": "integer", "default": 0, "minimum": 0},
            "color_correction": {"type": "string", "default": DEFAULT_COLOR_CORRECTION},
            "seed": {"type": "integer", "default": DEFAULT_SEED},
            "attention_mode": {"type": "string", "default": DEFAULT_ATTENTION_MODE},
            "blocks_to_swap": {"type": "integer", "default": 0, "minimum": 0},
            "dit_offload_device": {"type": "string", "default": "cpu"},
        },
    }
