from __future__ import annotations

from dataclasses import dataclass, field

from videox.engines.fields import (
    IMAGE,
    boolean_field,
    choice_notes,
    conditional,
    integer_field,
    number_field,
    schema,
    string_field,
)
from videox.errors import VideoXError

RELEASE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download"

DEFAULT_MODEL = "RealESRGAN_x4plus"
DENOISE_MODEL = "realesr-general-x4v3"
DEFAULT_OUTSCALE = 4
DEFAULT_DENOISE_STRENGTH = 0.5
DEFAULT_TILE = 512
DEFAULT_TILE_PAD = 10
DEFAULT_FP16 = True


@dataclass(frozen=True)
class WeightFile:
    name: str
    url: str


def _release(tag: str, name: str) -> WeightFile:
    return WeightFile(name, f"{RELEASE_URL}/{tag}/{name}")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    note: str
    arch: str
    scale: int
    weights: WeightFile
    arch_kwargs: dict[str, int | str] = field(default_factory=dict)
    # Second checkpoint blended in by denoise strength (deep network interpolation).
    denoise_weights: WeightFile | None = None

    @property
    def multiple_of(self) -> int:
        # RRDBNet below 4x pixel-unshuffles its input, so H and W must divide evenly.
        if self.arch == "rrdbnet" and self.scale < 4:
            return 4 // self.scale
        return 1

    @property
    def files(self) -> tuple[WeightFile, ...]:
        if self.denoise_weights is None:
            return (self.weights,)
        return (self.weights, self.denoise_weights)


def _rrdb(scale: int, num_block: int = 23) -> dict[str, int | str]:
    return {
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": num_block,
        "num_grow_ch": 32,
        "scale": scale,
    }


def _srvgg(num_conv: int) -> dict[str, int | str]:
    return {
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_conv": num_conv,
        "upscale": 4,
        "act_type": "prelu",
    }


MODELS: dict[str, ModelSpec] = {
    "RealESRGAN_x4plus": ModelSpec(
        label="通用照片 4 倍",
        note="照片和实拍画面的首选，细节锐利，偶尔会把纹理画得过头。",
        arch="rrdbnet",
        scale=4,
        weights=_release("v0.1.0", "RealESRGAN_x4plus.pth"),
        arch_kwargs=_rrdb(4),
    ),
    "RealESRNet_x4plus": ModelSpec(
        label="通用照片 4 倍（更平滑，少伪影）",
        note="不加对抗训练的版本，画面更平滑、伪影更少，细节也更软。",
        arch="rrdbnet",
        scale=4,
        weights=_release("v0.1.1", "RealESRNet_x4plus.pth"),
        arch_kwargs=_rrdb(4),
    ),
    "RealESRGAN_x2plus": ModelSpec(
        label="通用照片 2 倍",
        note="只想放大 2 倍时用，比 4 倍模型再缩小更快、更省显存。",
        arch="rrdbnet",
        scale=2,
        weights=_release("v0.2.1", "RealESRGAN_x2plus.pth"),
        arch_kwargs=_rrdb(2),
    ),
    "RealESRGAN_x4plus_anime_6B": ModelSpec(
        label="动漫插画 4 倍",
        note="动漫、插画和线稿，线条干净、色块平整，网络更小。",
        arch="rrdbnet",
        scale=4,
        weights=_release("v0.2.2.4", "RealESRGAN_x4plus_anime_6B.pth"),
        arch_kwargs=_rrdb(4, num_block=6),
    ),
    "realesr-general-x4v3": ModelSpec(
        label="轻量通用 4 倍（可调去噪）",
        note="速度最快的通用模型，能用去噪强度控制保留多少纹理。",
        arch="srvgg",
        scale=4,
        weights=_release("v0.2.5.0", "realesr-general-x4v3.pth"),
        arch_kwargs=_srvgg(32),
        denoise_weights=_release("v0.2.5.0", "realesr-general-wdn-x4v3.pth"),
    ),
    "realesr-animevideov3": ModelSpec(
        label="动漫轻量 4 倍",
        note="动漫截图和番剧画面的轻量模型，最快、最省显存，细节略少。",
        arch="srvgg",
        scale=4,
        weights=_release("v0.2.5.0", "realesr-animevideov3.pth"),
        arch_kwargs=_srvgg(16),
    ),
}


def model_spec(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise VideoXError(f"未知的 Real-ESRGAN 模型: {model_id}") from None


def required_weights(model_id: str, denoise_strength: float = DEFAULT_DENOISE_STRENGTH) -> tuple[WeightFile, ...]:
    """Files this run needs. Full denoise strength uses the x4v3 weights alone."""
    spec = model_spec(model_id)
    if spec.denoise_weights is None or denoise_strength >= 1:
        return (spec.weights,)
    return spec.files


def params_schema() -> dict:
    return schema(
        [("primary", "模型与倍数"), ("memory", "显存"), ("performance", "性能")],
        {
            "model": choice_notes(
                string_field(
                    "模型",
                    DEFAULT_MODEL,
                    group="primary",
                    media=IMAGE,
                    choices=[(model_id, spec.label) for model_id, spec in MODELS.items()],
                    description="决定用哪个超分网络。照片选通用照片 4 倍；想更平滑、少伪影选 RealESRNet；只要 2 倍选 2 倍模型；动漫和插画选动漫模型。轻量模型更快、更省显存，细节略少。首次选用会下载对应权重。",
                ),
                {model_id: spec.note for model_id, spec in MODELS.items()},
            ),
            "outscale": conditional(
                number_field(
                    "输出倍数",
                    DEFAULT_OUTSCALE,
                    group="primary",
                    media=IMAGE,
                    minimum=1,
                    maximum=8,
                    description="成品相对原图的放大倍数，范围 1 到 8。网络先按自身倍数放大，再缩放到这里的倍数。低于网络倍数时细节更扎实、文件更小；高于网络倍数只是插值，不会多出细节。默认跟随模型。",
                ),
                overrides=[({"model": ["RealESRGAN_x2plus"]}, {"default": 2})],
            ),
            "denoise_strength": conditional(
                number_field(
                    "去噪强度",
                    DEFAULT_DENOISE_STRENGTH,
                    group="primary",
                    media=IMAGE,
                    minimum=0,
                    maximum=1,
                    description="只对轻量通用模型生效，范围 0 到 1。调高去噪更强、画面更干净，也可能抹掉细小纹理；调低保留更多纹理和颗粒，噪点也会留下。默认 0.5。",
                ),
                visible_when={"model": [DENOISE_MODEL]},
            ),
            "tile": integer_field(
                "分块大小",
                DEFAULT_TILE,
                group="memory",
                media=IMAGE,
                minimum=0,
                maximum=4096,
                description="把大图切成这个边长的方块逐块处理，0 表示整张一次处理。调小更省显存，块数变多、速度变慢；调大或设为 0 更快，但大图容易爆显存。显存不足时先把它调小，例如 256。默认 512。",
            ),
            "tile_pad": integer_field(
                "分块重叠",
                DEFAULT_TILE_PAD,
                group="memory",
                media=IMAGE,
                minimum=0,
                maximum=128,
                description="每块向四周多读的像素，用来藏住块与块之间的接缝。调大接缝更不明显，重复计算更多、略慢；调小更快，接缝可能显出来。整张处理时不起作用。默认 10。",
            ),
            "fp16": boolean_field(
                "半精度",
                DEFAULT_FP16,
                group="performance",
                media=IMAGE,
                description="用半精度计算。打开后更快、更省显存，画质几乎不变；结果出现黑块或杂点时关掉，改用全精度，速度会慢一些。CPU 上始终用全精度，这一项不起作用。",
            ),
        },
    )
