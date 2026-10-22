"""Run FlashVSR v1.1 and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlashVSR 推理")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", default="tiny-long")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sparse-ratio", type=float, default=2.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--kv-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--color-fix", action="store_true")
    parser.add_argument("--no-color-fix", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    repo = Path(os.environ.get("VIDEOX_FLASHVSR_ROOT", ""))
    wan = repo / "examples" / "WanVSR"
    if wan.is_dir():
        sys.path.insert(0, str(wan))
    from videox.engines.flashvsr.infer import FlashRequest, run_flashvsr

    run_flashvsr(
        FlashRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_dir=Path(args.model_dir),
            mode=args.mode,
            scale=args.scale,
            sparse_ratio=args.sparse_ratio,
            local_range=args.local_range,
            kv_ratio=args.kv_ratio,
            seed=args.seed,
            color_fix=not args.no_color_fix,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        raise SystemExit(1) from exc
