from __future__ import annotations

from typing import ClassVar

from videox.engines.image_engine import MainImageEngine, format_number
from videox.engines.scunet.presets import (
    DEFAULT_MODEL,
    DEFAULT_STRENGTH,
    DEFAULT_TILE,
    DEFAULT_TILE_PAD,
    params_schema,
)
from videox.jobs.models import Capability


class SCUNetEngine(MainImageEngine):
    id = "scunet"
    name = "SCUNet"
    description = "真实照片盲去噪，适合夜景、暗光、老照片和压缩噪点，不用先判断噪点有多重。不改变图片尺寸。只使用一张显卡。"
    capabilities: ClassVar[list[Capability]] = ["denoise"]
    weights_folder = "SCUNet"
    entry_module = "videox.engines.scunet.progress_entry"

    def params_schema(self) -> dict:
        return params_schema()

    def entry_args(self, values: dict[str, object]) -> list[str]:
        return [
            "--model",
            str(values.get("model", DEFAULT_MODEL)),
            "--strength",
            format_number(values.get("strength", DEFAULT_STRENGTH)),
            "--tile",
            str(values.get("tile", DEFAULT_TILE)),
            "--tile-pad",
            str(values.get("tile_pad", DEFAULT_TILE_PAD)),
        ]
