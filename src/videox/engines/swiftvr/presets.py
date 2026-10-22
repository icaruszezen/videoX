from __future__ import annotations

from videox.engines.fields import integer_field, schema, string_field

DEFAULT_UPSCALE = 4
DEFAULT_CLIP_LEN = 24
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SHORT_SIDE = 0

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
                description="输出尺寸＝原片×倍数。整帧一次处理、不分块，显存随输出像素增长：官方在 H100 上输出 2560×1440 峰值约 38 GB，32 GB 显卡大约只能输出 1080p。720p 及以上的原片请用 1–2 倍，或设置“输出短边”，否则会显存不足。",
            ),
            "short_side": integer_field(
                "输出短边",
                DEFAULT_SHORT_SIDE,
                group="primary",
                minimum=0,
                maximum=4320,
                description="设定后忽略放大倍数，按原片比例把输出短边缩放到这个像素数，例如 1080。0 表示按放大倍数输出。",
            ),
            "clip_len": integer_field(
                "分块长度",
                DEFAULT_CLIP_LEN,
                group="primary",
                minimum=4,
                maximum=64,
                description="中间块的帧数，必须是 4 的倍数。调大时时间更连贯，显存占用也大致按比例增加；调小更快、更省显存（24 改成 8 约省到三分之一）。默认 24。",
                step=4,
            ),
            "dtype": string_field(
                "精度",
                DEFAULT_DTYPE,
                group="primary",
                choices=DTYPE_CHOICES,
                description="bfloat16 是默认精度，速度和画质更稳。30 系（Ampere）之前的显卡没有原生 bfloat16，会自动改用 float16。",
            ),
        },
    )
