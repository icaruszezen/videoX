from __future__ import annotations

from typing import ClassVar

from videox.engines.image_engine import MainImageEngine, format_number
from videox.engines.realesrgan.presets import params_schema
from videox.jobs.models import Capability


class RealESRGANEngine(MainImageEngine):
    id = "realesrgan"
    name = "Real-ESRGAN"
    description = "GAN 图片超分，速度快、显存占用低，适合照片、插画和动漫，提供 2 倍和 4 倍模型。只使用一张显卡。"
    capabilities: ClassVar[list[Capability]] = ["upscale"]
    weights_folder = "RealESRGAN"
    entry_module = "videox.engines.realesrgan.progress_entry"

    def params_schema(self) -> dict:
        return params_schema()

    def entry_args(self, values: dict[str, object]) -> list[str]:
        args = [
            "--model",
            str(values["model"]),
            "--outscale",
            format_number(values["outscale"]),
        ]
        # Only present when the model uses it; hidden fields are dropped upstream.
        if "denoise_strength" in values:
            args += ["--denoise-strength", format_number(values["denoise_strength"])]
        args += [
            "--tile",
            str(values["tile"]),
            "--tile-pad",
            str(values["tile_pad"]),
            "--fp16" if values["fp16"] else "--no-fp16",
        ]
        return args
