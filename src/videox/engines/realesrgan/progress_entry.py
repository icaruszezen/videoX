"""Run Real-ESRGAN on one image and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from videox.engines.realesrgan.presets import (
    DEFAULT_DENOISE_STRENGTH,
    DEFAULT_FP16,
    DEFAULT_MODEL,
    DEFAULT_TILE,
    DEFAULT_TILE_PAD,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-ESRGAN 图片超分")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # Leaving --outscale out keeps the model's native scale.
    parser.add_argument("--outscale", type=float, default=None)
    parser.add_argument("--denoise-strength", type=float, default=DEFAULT_DENOISE_STRENGTH)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--tile-pad", type=int, default=DEFAULT_TILE_PAD)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=DEFAULT_FP16)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from videox.engines.realesrgan.infer import RealESRGANRequest, run_realesrgan

    run_realesrgan(
        RealESRGANRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_dir=Path(args.model_dir),
            model_id=args.model,
            outscale=args.outscale,
            denoise_strength=args.denoise_strength,
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
