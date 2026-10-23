from __future__ import annotations

from videox.engines.fields import boolean_field, integer_field, number_field, schema, string_field

DEFAULT_MODE = "tiny-long"
DEFAULT_SCALE = "4"
DEFAULT_SPARSE_RATIO = 2.0
DEFAULT_LOCAL_RANGE = "11"
DEFAULT_KV_RATIO = 3
DEFAULT_SEED = 0

WEIGHT_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
    "Wan2.1_VAE.pth",
)
WEIGHT_REPO = "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main/"


def weights_for(mode: str) -> tuple[str, ...]:
    """Tiny modes decode with TCDecoder; only Full needs the 0.5 GB Wan VAE."""
    base = ("diffusion_pytorch_model_streaming_dmd.safetensors", "LQ_proj_in.ckpt")
    return base + (("Wan2.1_VAE.pth",) if mode == "full" else ("TCDecoder.ckpt",))

MODE_CHOICES = [
    ("tiny-long", "Tiny Long"),
    ("full", "Full"),
]
# Requests from before Tiny and Tiny Long ran the same loop.
MODE_ALIASES = {"tiny": "tiny-long"}
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
                description="两种模式都每 8 帧一段，边读边处理边写出，内存和显存不随视频时长增长，一小时的视频也能处理。Tiny Long 用轻量解码器，更快、更省显存，是默认选择。Full 用 Wan VAE 解码，细节更多，但更慢，解码缓存还要一直占着显存，输出 1080p 时比 Tiny Long 多用约 10 GB；第一次使用还要多下载约 0.5 GB 权重。"
            ),
            "scale": string_field(
                "放大倍数",
                DEFAULT_SCALE,
                group="primary",
                choices=SCALE_CHOICES,
                description="输出相对原片的倍数，宽高比不变。模型按 4 倍训练，4 倍最稳。显存随输出的宽×高增长，2 倍的输出只有 4 倍的四分之一大，更快、更省显存：原片 480p 时，Tiny Long 4 倍估计要 40 GB 左右显存，2 倍约 13–16 GB。开始处理时会打印这次的显存估计。",
            ),
            "sparse_ratio": number_field(
                "稀疏比",
                DEFAULT_SPARSE_RATIO,
                group="quality",
                minimum=0.5,
                maximum=5,
                description="控制稀疏注意力的密度。1.5 更快，2.0 更稳。",
            ),
            "local_range": string_field(
                "局部范围",
                DEFAULT_LOCAL_RANGE,
                group="quality",
                choices=LOCAL_RANGE_CHOICES,
                description="局部注意力的窗口。11 更稳，9 边缘更锐。",
            ),
            "kv_ratio": integer_field(
                "KV 比",
                DEFAULT_KV_RATIO,
                group="quality",
                minimum=2,
                maximum=8,
                description="每步保留多少段历史（每段约 8 帧）作为参考。这些历史放在显存里，输出 1080p 时 KV 比每加 1 多占约 3 GB。显存不够时先调到 2；调高更不容易丢前面的纹理。2 是最低有效值，默认 3。",
            ),
            "color_fix": boolean_field(
                "色彩校正",
                True,
                group="quality",
                description="按每帧每个颜色通道的均值和标准差（AdaIN）把输出颜色对齐原片。打开后整体色调更接近原片；关掉会让模型自己决定颜色，可能整体偏色。",
            ),
            "seed": integer_field(
                "随机种子",
                DEFAULT_SEED,
                group="quality",
                description="相同种子和相同输入会得到相同结果。换一个数字只改变少量纹理。",
            ),
        },
    )
