from __future__ import annotations

from dataclasses import dataclass

from videox.engines.fields import (
    IMAGE,
    choice_notes,
    integer_field,
    number_field,
    schema,
    string_field,
)
from videox.errors import VideoXError

RELEASE_URL = "https://github.com/cszn/KAIR/releases/download/v1.0"

DEFAULT_MODEL = "real_psnr"
DEFAULT_STRENGTH = 1.0
DEFAULT_TILE = 512
DEFAULT_TILE_PAD = 32
MAX_TILE = 4096
MAX_TILE_PAD = 256


@dataclass(frozen=True)
class SCUNetModel:
    """One released checkpoint and the network settings it was trained with.

    The settings follow upstream ``main_test_scunet_real_application.py``:
    ``SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)``. The files are
    plain state dicts. SCUNet keeps the input size, so ``scale`` is always 1.
    """

    id: str
    label: str
    note: str
    file: str
    scale: int = 1
    in_nc: int = 3
    config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4)
    dim: int = 64

    @property
    def files(self) -> tuple[str, ...]:
        return (self.file,)

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(f"{RELEASE_URL}/{name}" for name in self.files)

    def arch_kwargs(self) -> dict[str, object]:
        return {"in_nc": self.in_nc, "config": list(self.config), "dim": self.dim}


MODELS: dict[str, SCUNetModel] = {
    model.id: model
    for model in (
        SCUNetModel(
            id="real_psnr",
            label="去噪（保真）",
            note="常用选择：去掉噪点和压缩痕迹，颜色和结构最贴近原图，细节偏柔和。",
            file="scunet_color_real_psnr.pth",
        ),
        SCUNetModel(
            id="real_gan",
            label="去噪（更锐）",
            note="边缘和纹理更清楚、更有质感；可能补出原图没有的细碎纹理，人脸和文字要留意。",
            file="scunet_color_real_gan.pth",
        ),
    )
}


def model_spec(model_id: str) -> SCUNetModel:
    try:
        return MODELS[model_id]
    except KeyError:
        raise VideoXError(f"未知的 SCUNet 模型: {model_id}") from None


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
                    description="两种都是盲去噪，不用先判断噪点有多重，也不放大，输出和原图一样大。保真版改动最少、最稳，画面偏柔和；更锐版边缘和纹理更清楚，但可能补出原图没有的细节。首次选用会下载对应权重。",
                ),
                {model.id: model.note for model in MODELS.values()},
            ),
            "strength": number_field(
                "去噪强度",
                DEFAULT_STRENGTH,
                group="primary",
                media=IMAGE,
                minimum=0,
                maximum=1,
                description="把去噪结果和原图按比例混合，范围 0 到 1。1 完全用去噪结果，最干净；调低会掺回原图，保留更多原始颗粒和细节，噪点也跟着回来一部分，0 等于原图。画面被抹得发糊、发假时调到 0.6 到 0.8。默认 1。",
            ),
            "tile": integer_field(
                "分块大小",
                DEFAULT_TILE,
                group="memory",
                media=IMAGE,
                minimum=0,
                maximum=MAX_TILE,
                description="把图片切成这个边长的方块逐块处理。调小更省显存，但块数更多、更慢，太小时每块看到的范围太少，去噪效果会变差；调大略快，显存按面积增长，不够会失败。0 表示整张一次处理，只适合小图或大显存。块会自动补齐到 64 的倍数，分块大小加两倍重叠正好是 64 的倍数时最省计算。默认 512。",
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
