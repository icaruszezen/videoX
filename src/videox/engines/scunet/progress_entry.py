"""Run SCUNet on one image and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from videox.engines.scunet.presets import DEFAULT_MODEL, DEFAULT_STRENGTH, DEFAULT_TILE, DEFAULT_TILE_PAD


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SCUNet 推理")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--tile-pad", type=int, default=DEFAULT_TILE_PAD)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from videox.engines.scunet.infer import SCUNetRequest, run_scunet

    run_scunet(
        SCUNetRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_dir=Path(args.model_dir),
            model=args.model,
            strength=args.strength,
            tile=args.tile,
            tile_pad=args.tile_pad,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        raise SystemExit(1) from exc
