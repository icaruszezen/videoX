from __future__ import annotations

from videox.engines.fields import boolean_field, integer_field, number_field, schema, string_field

DEFAULT_MODE = "tiny-long"
DEFAULT_SCALE = "4"
DEFAULT_SPARSE_RATIO = 2.0
DEFAULT_LOCAL_RANGE = "11"
DEFAULT_KV_RATIO = 3.0
DEFAULT_SEED = 0

WEIGHT_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
    "Wan2.1_VAE.pth",
)
WEIGHT_REPO = "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main/"

MODE_CHOICES = [
    ("tiny-long", "Tiny Long（长视频）"),
    ("tiny", "Tiny"),
    ("full", "Full"),
]
SCALE_CHOICES = [("4", "4 倍"), ("2", "2 倍")]
LOCAL_RANGE_CHOICES = [("11", "11（更稳）"), ("9", "9（更锐）")]


def params_schema() -> dict:
    return schema(
        [("primary", "模型"), ("quality", "画质")],
        {
            "mode": string_field(
                "模式",
                DEFAULT_MODE,
                group="primary",
                choices=MODE_CHOICES,
                description="Tiny Long 按流式方式处理长视频，显存更低，是长片的默认。Tiny 更快，适合较短的视频。Full 细节更多，显存和耗时都更高。",
            ),
            "scale": string_field(
                "放大倍数",
                DEFAULT_SCALE,
                group="primary",
                choices=SCALE_CHOICES,
                description="输出相对原片的倍数。模型按 4 倍训练，4 倍最稳。2 倍更快，适合原片已经比较大的时候。",
            ),
            "sparse_ratio": number_field(
                "稀疏比",
                DEFAULT_SPARSE_RATIO,
                group="quality",
                minimum=0.5,
                maximum=5,
                description="控制稀疏注意力的密度。1.5 更快，2.0 更稳。没有稀疏内核时这项不生效，会改用整块注意力。",
            ),
            "local_range": string_field(
                "局部范围",
                DEFAULT_LOCAL_RANGE,
                group="quality",
                choices=LOCAL_RANGE_CHOICES,
                description="局部注意力的窗口。11 更稳，9 边缘更锐。",
            ),
            "kv_ratio": number_field(
                "KV 比",
                DEFAULT_KV_RATIO,
                group="quality",
                minimum=1,
                maximum=8,
                description="保留多少历史键值。调低更省显存，调高更不容易丢前面的纹理。默认 3。",
            ),
            "color_fix": boolean_field(
                "色彩校正",
                True,
                group="quality",
                description="用小波把输出颜色拉回原片。打开后更接近原色。关掉会让模型自己决定颜色，可能偏色。",
            ),
            "seed": integer_field(
                "随机种子",
                DEFAULT_SEED,
                group="quality",
                description="相同种子和相同输入会得到相同结果。换一个数字只改变少量纹理。",
            ),
        },
    )
