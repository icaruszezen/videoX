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
                description="Tiny Long 按流式方式逐段送进显卡，显存占用最低，是默认选择；但整段视频放大后的画面仍会放在内存里，内存占用随时长和输出分辨率增长（例如 480p 放大 4 倍，每秒视频约需 3–4 GB 内存），长视频请先剪短或改用 2 倍。Tiny 更快，Full 细节更多，但这两种会把整段放大后的视频放进显存，只适合几秒的短片。"
            ),
            "scale": string_field(
                "放大倍数",
                DEFAULT_SCALE,
                group="primary",
                choices=SCALE_CHOICES,
                description="输出相对原片的倍数，宽高比不变。模型按 4 倍训练，4 倍最稳。2 倍更快、更省内存，适合原片已经比较大的时候。",
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
                description="每步保留多少段历史（每段约 8 帧）作为参考。调低更省显存，调高更不容易丢前面的纹理。2 是最低有效值，默认 3。",
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
