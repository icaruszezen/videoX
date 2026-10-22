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

GFPGAN_RELEASE_URL = "https://github.com/TencentARC/GFPGAN/releases/download"
FACEXLIB_RELEASE_URL = "https://github.com/xinntao/facexlib/releases/download"

DEFAULT_VERSION = "v1.4"
DEFAULT_UPSCALE = "2"
DEFAULT_BACKGROUND = "RealESRGAN_x2plus"
NO_BACKGROUND = "none"
DEFAULT_STRENGTH = 1.0
DEFAULT_ONLY_CENTER_FACE = False
DEFAULT_TILE = 512
DEFAULT_TILE_PAD = 10
DEFAULT_FP16 = True

# Faces are aligned to and restored at this size, as upstream.
FACE_SIZE = 512
# Faces whose eyes are closer than this many input pixels are skipped (upstream value).
EYE_DIST_THRESHOLD = 5
# facexlib detects at the input size; larger inputs are detected on a smaller copy.
MAX_DETECT_EDGE = 4096
DETECT_CONFIDENCE = 0.97

# The sibling weights folder Real-ESRGAN downloads into.
BACKGROUND_WEIGHTS_FOLDER = "RealESRGAN"


@dataclass(frozen=True)
class WeightFile:
    name: str
    url: str


DETECTOR = WeightFile(
    "detection_Resnet50_Final.pth",
    f"{FACEXLIB_RELEASE_URL}/v0.1.0/detection_Resnet50_Final.pth",
)


def _clean_arch() -> dict[str, int | float | bool]:
    # GFPGANer(arch="clean", channel_multiplier=2) in upstream gfpgan/utils.py.
    return {
        "out_size": FACE_SIZE,
        "num_style_feat": 512,
        "channel_multiplier": 2,
        "num_mlp": 8,
        "input_is_latent": True,
        "different_w": True,
        "narrow": 1,
        "sft_half": True,
    }


@dataclass(frozen=True)
class GFPGANModel:
    label: str
    note: str
    weights: WeightFile
    arch_kwargs: dict[str, int | float | bool] = field(default_factory=_clean_arch)

    @property
    def files(self) -> tuple[WeightFile, ...]:
        """The restorer first, then the face detector every version needs."""
        return (self.weights, DETECTOR)


MODELS: dict[str, GFPGANModel] = {
    "v1.4": GFPGANModel(
        label="v1.4（细节更多）",
        note="细节更多、五官更清楚，对很糊的脸提升更明显，偶尔显得过锐。",
        weights=WeightFile("GFPGANv1.4.pth", f"{GFPGAN_RELEASE_URL}/v1.3.0/GFPGANv1.4.pth"),
    ),
    "v1.3": GFPGANModel(
        label="v1.3（更自然）",
        note="结果更自然，过度锐化和涂抹感更少，适合本身不太糊的人像。",
        weights=WeightFile("GFPGANv1.3.pth", f"{GFPGAN_RELEASE_URL}/v1.3.0/GFPGANv1.3.pth"),
    ),
}

UPSCALE_CHOICES = [("1", "1 倍（不放大）"), ("2", "2 倍"), ("4", "4 倍")]

# Background id -> (label, note). Every id except "none" is a Real-ESRGAN model id.
BACKGROUNDS: dict[str, tuple[str, str]] = {
    "RealESRGAN_x2plus": (
        "Real-ESRGAN 2 倍",
        "和官方默认一致。2 倍网络速度快、显存占用低；放大 4 倍时再用插值补足，背景略软。",
    ),
    "RealESRGAN_x4plus": (
        "Real-ESRGAN 4 倍",
        "4 倍网络，放大 4 倍时背景更清楚；放大 1 或 2 倍时更慢，收益不大。",
    ),
    NO_BACKGROUND: (
        "只修人脸",
        "背景用普通插值放大，最快、最省显存，背景不会变清楚。",
    ),
}
BACKGROUND_UPSCALERS = [key for key in BACKGROUNDS if key != NO_BACKGROUND]


def model_spec(version: str) -> GFPGANModel:
    try:
        return MODELS[version]
    except KeyError:
        raise VideoXError(f"未知的 GFPGAN 版本: {version}") from None


def params_schema() -> dict:
    with_background = {"background": list(BACKGROUND_UPSCALERS)}
    return schema(
        [("primary", "模型与倍数"), ("quality", "人脸"), ("memory", "显存"), ("performance", "性能")],
        {
            "version": choice_notes(
                string_field(
                    "模型版本",
                    DEFAULT_VERSION,
                    group="primary",
                    media=IMAGE,
                    choices=[(version, spec.label) for version, spec in MODELS.items()],
                    description="决定用哪一版 GFPGAN 修脸。v1.4 细节更多、更清楚，偶尔显得过锐；v1.3 更自然，过锐和涂抹感更少，细节略少。首次选用会下载对应权重（约 350MB）和人脸检测模型。",
                ),
                {version: spec.note for version, spec in MODELS.items()},
            ),
            "upscale": string_field(
                "放大倍数",
                DEFAULT_UPSCALE,
                group="primary",
                media=IMAGE,
                choices=UPSCALE_CHOICES,
                description="成品相对原图的放大倍数。人脸先修复到 512 像素再贴回，倍数越高，修出来的五官细节保留得越多；背景按背景处理的设置放大。调高画面更大，处理更慢、更占内存；选 1 倍只修脸不放大。默认 2。",
            ),
            "background": conditional(
                choice_notes(
                    string_field(
                        "背景处理",
                        DEFAULT_BACKGROUND,
                        group="primary",
                        media=IMAGE,
                        choices=[(key, label) for key, (label, _note) in BACKGROUNDS.items()],
                        description="人脸以外的画面怎么放大。交给 Real-ESRGAN 时背景也会去噪变清楚，但更慢、更吃显存；只修人脸时背景用普通插值放大，最快，背景偏软。放大 4 倍时默认换成 4 倍模型。首次选用会下载对应的 Real-ESRGAN 权重。",
                    ),
                    {key: note for key, (_label, note) in BACKGROUNDS.items()},
                ),
                overrides=[({"upscale": ["4"]}, {"default": "RealESRGAN_x4plus"})],
            ),
            "strength": number_field(
                "人脸修复强度",
                DEFAULT_STRENGTH,
                group="quality",
                media=IMAGE,
                minimum=0,
                maximum=1,
                description="修复后的人脸盖在放大后背景上的比例，范围 0 到 1。1 完全用修复后的人脸，最干净清楚；调低会透出更多按“背景处理”放大的原图人脸，保留原来的五官和皮肤纹理，更像本人，但模糊和噪点也会留下一些；0 等于不修脸，也不运行人脸模型。默认 1。",
            ),
            "only_center_face": boolean_field(
                "只修中间的人脸",
                DEFAULT_ONLY_CENTER_FACE,
                group="quality",
                media=IMAGE,
                description="打开后只修离画面中心最近的一张脸，其他人脸只随背景放大。适合单人照里有路人、或者把图案误认成人脸的情况。关闭时修复检测到的所有人脸，人多时更慢。默认关闭。",
            ),
            "tile": conditional(
                integer_field(
                    "背景分块大小",
                    DEFAULT_TILE,
                    group="memory",
                    media=IMAGE,
                    minimum=0,
                    maximum=4096,
                    description="背景交给 Real-ESRGAN 时，把大图切成这个边长的方块逐块处理，0 表示整张一次处理。调小更省显存，块数变多、速度变慢；调大或设为 0 更快，但大图容易爆显存。显存不足时先把它调小，例如 256。默认 512。",
                ),
                visible_when=with_background,
            ),
            "tile_pad": conditional(
                integer_field(
                    "背景分块重叠",
                    DEFAULT_TILE_PAD,
                    group="memory",
                    media=IMAGE,
                    minimum=0,
                    maximum=128,
                    description="背景每块向四周多读的像素，用来藏住块与块之间的接缝。调大接缝更不明显，重复计算更多、略慢；调小更快，接缝可能显出来。整张处理时不起作用。默认 10。",
                ),
                visible_when=with_background,
            ),
            "fp16": conditional(
                boolean_field(
                    "背景半精度",
                    DEFAULT_FP16,
                    group="performance",
                    media=IMAGE,
                    description="背景网络用半精度计算。打开后更快、更省显存，画质几乎不变；背景出现黑块或杂点时关掉，改用全精度，速度会慢一些。人脸网络始终用全精度。CPU 上始终用全精度，这一项不起作用。",
                ),
                visible_when=with_background,
            ),
        },
    )
