"""Run GFPGAN on one image and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from videox.engines.gfpgan.presets import (
    DEFAULT_BACKGROUND,
    DEFAULT_FP16,
    DEFAULT_ONLY_CENTER_FACE,
    DEFAULT_STRENGTH,
    DEFAULT_TILE,
    DEFAULT_TILE_PAD,
    DEFAULT_UPSCALE,
    DEFAULT_VERSION,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GFPGAN 人脸修复")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--upscale", type=int, choices=[1, 2, 4], default=int(DEFAULT_UPSCALE))
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--only-center-face", action=argparse.BooleanOptionalAction, default=DEFAULT_ONLY_CENTER_FACE)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--tile-pad", type=int, default=DEFAULT_TILE_PAD)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=DEFAULT_FP16)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from videox.engines.gfpgan.infer import GFPGANRequest, run_gfpgan

    run_gfpgan(
        GFPGANRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_dir=Path(args.model_dir),
            version=args.version,
            upscale=args.upscale,
            background=args.background,
            strength=args.strength,
            only_center_face=args.only_center_face,
            tile=args.tile,
            tile_pad=args.tile_pad,
            half=args.fp16,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        raise SystemExit(1) from exc
