from __future__ import annotations

from videox.engines.fields import integer_field, schema, string_field

DEFAULT_UPSCALE = 4
DEFAULT_CLIP_LEN = 24
DEFAULT_DTYPE = "bfloat16"

WEIGHT_FILES = (
    "reae.safetensors",
    "prompt_embedding.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
)
WEIGHT_REPO = "https://huggingface.co/H-oliday/SwiftVR/resolve/main/"

DTYPE_CHOICES = [("bfloat16", "bfloat16"), ("float16", "float16")]


def params_schema() -> dict:
    return schema(
        [("primary", "模型")],
        {
            "upscale": integer_field(
                "放大倍数",
                DEFAULT_UPSCALE,
                group="primary",
                minimum=1,
                maximum=4,
                description="输出相对原片的倍数。官方按 4 倍使用。调低会少算一些像素，速度更快，清晰度也少一档。",
            ),
            "clip_len": integer_field(
                "分块长度",
                DEFAULT_CLIP_LEN,
                group="primary",
                minimum=4,
                maximum=64,
                description="中间块的帧数，必须是 4 的倍数。调大时时间更连贯，也更吃显存；调小更快、更省显存。默认 24。",
            ),
            "dtype": string_field(
                "精度",
                DEFAULT_DTYPE,
                group="primary",
                choices=DTYPE_CHOICES,
                description="bfloat16 是默认精度，速度和画质更稳。显卡不支持时改成 float16。",
            ),
        },
    )
