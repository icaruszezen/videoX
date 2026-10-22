"""Run SwiftVR and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SwiftVR 推理")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=24)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from videox.engines.swiftvr.infer import SwiftRequest, run_swiftvr

    run_swiftvr(
        SwiftRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            checkpoint=Path(args.checkpoint),
            upscale=args.upscale,
            clip_len=args.clip_len,
            dtype=args.dtype,
            short_side=args.short_side,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        raise SystemExit(1) from exc
