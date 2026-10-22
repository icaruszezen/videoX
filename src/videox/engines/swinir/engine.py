from __future__ import annotations

from typing import ClassVar

from videox.engines.image_engine import MainImageEngine, format_number
from videox.engines.swinir.presets import DEFAULT_MODEL, DEFAULT_TILE, DEFAULT_TILE_PAD, params_schema
from videox.jobs.models import Capability


class SwinIREngine(MainImageEngine):
    id = "swinir"
    name = "SwinIR"
    description = "基于 Swin Transformer 的图片修复，可做真实照片超分、去噪和去 JPEG 压缩痕迹。比 GAN 类模型更慢、更吃显存，细节更自然。只使用一张显卡。"
    capabilities: ClassVar[list[Capability]] = ["upscale", "denoise"]
    weights_folder = "SwinIR"
    entry_module = "videox.engines.swinir.progress_entry"

    def params_schema(self) -> dict:
        return params_schema()

    def entry_args(self, values: dict[str, object]) -> list[str]:
        args = ["--model", str(values.get("model", DEFAULT_MODEL))]
        # Only the super-resolution models show an output scale; the others keep the input size.
        if "outscale" in values:
            args += ["--outscale", format_number(values["outscale"])]
        args += [
            "--tile",
            str(values.get("tile", DEFAULT_TILE)),
            "--tile-pad",
            str(values.get("tile_pad", DEFAULT_TILE_PAD)),
        ]
        return args
