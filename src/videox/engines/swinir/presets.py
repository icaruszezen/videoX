from __future__ import annotations

from dataclasses import dataclass

from videox.engines.fields import (
    IMAGE,
    choice_notes,
    conditional,
    integer_field,
    number_field,
    schema,
    string_field,
)
from videox.errors import VideoXError

RELEASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"

DEFAULT_MODEL = "real_x4_m"
DEFAULT_OUTSCALE = 4
DEFAULT_TILE = 384
DEFAULT_TILE_PAD = 32
MAX_OUTSCALE = 8
MAX_TILE = 2048
MAX_TILE_PAD = 256


@dataclass(frozen=True)
class SwinIRModel:
    """One released checkpoint and the network settings it was trained with.

    The settings follow ``define_model`` in upstream ``main_test_swinir.py``.
    ``param_key`` is the checkpoint entry upstream loads; files without it are
    plain state dicts.
    """

    id: str
    label: str
    note: str
    file: str
    scale: int
    img_size: int
    window_size: int
    img_range: float
    depths: tuple[int, ...]
    embed_dim: int
    num_heads: tuple[int, ...]
    upsampler: str
    resi_connection: str
    param_key: str
    mlp_ratio: float = 2.0

    @property
    def files(self) -> tuple[str, ...]:
        return (self.file,)

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(f"{RELEASE_URL}/{name}" for name in self.files)

    def arch_kwargs(self) -> dict[str, object]:
        return {
            "upscale": self.scale,
            "in_chans": 3,
            "img_size": self.img_size,
            "window_size": self.window_size,
            "img_range": self.img_range,
            "depths": list(self.depths),
            "embed_dim": self.embed_dim,
            "num_heads": list(self.num_heads),
            "mlp_ratio": self.mlp_ratio,
            "upsampler": self.upsampler,
            "resi_connection": self.resi_connection,
        }


def _real(model_id: str, label: str, note: str, file: str, scale: int, large: bool = False) -> SwinIRModel:
    return SwinIRModel(
        id=model_id,
        label=label,
        note=note,
        file=file,
        scale=scale,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=(6,) * (9 if large else 6),
        embed_dim=240 if large else 180,
        num_heads=(8,) * 9 if large else (6,) * 6,
        upsampler="nearest+conv",
        resi_connection="3conv" if large else "1conv",
        param_key="params_ema",
    )


def _denoise(level: int, label: str, note: str) -> SwinIRModel:
    return SwinIRModel(
        id=f"denoise_{level}",
        label=label,
        note=note,
        file=f"005_colorDN_DFWB_s128w8_SwinIR-M_noise{level}.pth",
        scale=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=(6,) * 6,
        embed_dim=180,
        num_heads=(6,) * 6,
        upsampler="",
        resi_connection="1conv",
        param_key="params",
    )


def _jpeg(quality: int, note: str) -> SwinIRModel:
    # Window 7 lines up with JPEG's 8x8 blocks the way upstream trained it; img_range 255 as upstream.
    return SwinIRModel(
        id=f"jpeg_{quality}",
        label=f"去 JPEG 压缩（质量 {quality}）",
        note=note,
        file=f"006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg{quality}.pth",
        scale=1,
        img_size=126,
        window_size=7,
        img_range=255.0,
        depths=(6,) * 6,
        embed_dim=180,
        num_heads=(6,) * 6,
        upsampler="",
        resi_connection="1conv",
        param_key="params",
    )


MODELS: dict[str, SwinIRModel] = {
    model.id: model
    for model in (
        _real(
            "real_x4_m",
            "真实照片 4 倍（M）",
            "常用选择：模糊、低清的手机照片、网图和老照片放大 4 倍，速度和显存最均衡。",
            "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            4,
        ),
        _real(
            "real_x4_l",
            "真实照片 4 倍（L，细节更多）",
            "更大的网络，纹理和细节更丰富；明显更慢、更吃显存，显存紧张时把分块调小。",
            "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            4,
            large=True,
        ),
        _real(
            "real_x2_m",
            "真实照片 2 倍（M）",
            "原图已经比较清楚、只想放大一倍时用，改动更少、更接近原图。",
            "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",
            2,
        ),
        _denoise(15, "去噪（轻）", "轻微噪点，例如光线尚可的手机照片；保留细节最多。"),
        _denoise(25, "去噪（中）", "明显噪点，例如室内或傍晚拍的照片。"),
        _denoise(50, "去噪（重）", "噪点很重的暗光照片；去得最干净，细节也会抹掉一些。"),
        _jpeg(40, "轻度压缩：只有轻微毛刺；保留细节最多。"),
        _jpeg(30, "中等压缩：边缘有毛刺和蚊状噪点。"),
        _jpeg(20, "压缩较重：方块明显，文字和边缘周围有杂纹。"),
        _jpeg(10, "压缩极重：满是方块和色块，例如被反复转发的截图。"),
    )
}

SR_MODELS = [model_id for model_id, model in MODELS.items() if model.scale > 1]


def model_spec(model_id: str) -> SwinIRModel:
    try:
        return MODELS[model_id]
    except KeyError:
        raise VideoXError(f"未知的 SwinIR 模型: {model_id}") from None


def params_schema() -> dict:
    return schema(
        [("primary", "模型"), ("memory", "显存")],
        {
            "model": choice_notes(
                string_field(
                    "模型",
                    DEFAULT_MODEL,
                    group="primary",
                    media=IMAGE,
                    choices=[(model.id, model.label) for model in MODELS.values()],
                    description="决定做哪种修复。真实照片超分把模糊、低清的照片放大：M 速度和显存更均衡，L 细节更多，但更慢、更吃显存。去噪和去 JPEG 压缩不放大，输出和原图一样大。去噪按噪点轻重选轻、中、重；去 JPEG 压缩按原图压得多狠来选，质量数字越小清理越狠。清理越狠抹掉的细节越多，选刚好够用的一档。首次选用会下载对应权重。",
                ),
                {model.id: model.note for model in MODELS.values()},
            ),
            "outscale": conditional(
                number_field(
                    "输出倍数",
                    DEFAULT_OUTSCALE,
                    group="primary",
                    media=IMAGE,
                    minimum=1,
                    maximum=MAX_OUTSCALE,
                    description="成品相对原图的放大倍数。和模型自带倍数相同时直接输出；不同时先按模型倍数放大，再用 Lanczos 缩放到这里的倍数。调高画面更大，但超出模型倍数的部分只是插值，不会多出细节；调低文件更小、更省内存。",
                ),
                visible_when={"model": SR_MODELS},
                overrides=[
                    ({"model": [model_id]}, {"default": model.scale})
                    for model_id, model in MODELS.items()
                    if model.scale > 1 and model.scale != DEFAULT_OUTSCALE
                ],
            ),
            "tile": integer_field(
                "分块大小",
                DEFAULT_TILE,
                group="memory",
                media=IMAGE,
                minimum=0,
                maximum=MAX_TILE,
                description="把图片切成这个边长的方块逐块处理。SwinIR 的注意力很吃显存，调小更省显存，但块数更多、更慢，太小时相邻块的纹理可能对不上；调大略快，显存按面积增长，不够会失败。L 模型更吃显存，显存紧张时用 256 或更小。0 表示整张一次处理，只适合小图或大显存。块会自动补齐到模型窗口的倍数，填任意数值都可以。默认 384。",
            ),
            "tile_pad": integer_field(
                "分块重叠",
                DEFAULT_TILE_PAD,
                group="memory",
                media=IMAGE,
                minimum=0,
                maximum=MAX_TILE_PAD,
                description="每块向四周多读的像素，处理完只保留中间部分，用来藏住接缝。调大接缝更不明显，但重复计算更多、更慢、更占显存；调小更快，块与块之间可能露出接缝。分块大小为 0 时不起作用。默认 32。",
            ),
        },
    )
