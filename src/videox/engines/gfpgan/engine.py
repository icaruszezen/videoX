from __future__ import annotations

from typing import ClassVar

from videox.engines.gfpgan.presets import params_schema
from videox.engines.image_engine import MainImageEngine, format_number
from videox.jobs.models import Capability


class GFPGANEngine(MainImageEngine):
    id = "gfpgan"
    name = "GFPGAN"
    description = "检测并修复照片里的人脸，适合老照片和低清人像，背景可以交给 Real-ESRGAN 一起放大。只使用一张显卡。"
    capabilities: ClassVar[list[Capability]] = ["face", "upscale"]
    weights_folder = "GFPGAN"
    entry_module = "videox.engines.gfpgan.progress_entry"

    def params_schema(self) -> dict:
        return params_schema()

    def entry_args(self, values: dict[str, object]) -> list[str]:
        args = [
            "--version",
            str(values["version"]),
            "--upscale",
            str(values["upscale"]),
            "--background",
            str(values["background"]),
            "--strength",
            format_number(values["strength"]),
            "--only-center-face" if values["only_center_face"] else "--no-only-center-face",
        ]
        # The background settings are hidden (and absent) when only faces are restored.
        if "tile" in values:
            args += ["--tile", str(values["tile"])]
        if "tile_pad" in values:
            args += ["--tile-pad", str(values["tile_pad"])]
        if "fp16" in values:
            args.append("--fp16" if values["fp16"] else "--no-fp16")
        return args
