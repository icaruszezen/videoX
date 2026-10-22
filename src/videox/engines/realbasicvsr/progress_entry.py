"""Run RealBasicVSR and emit one JSON progress object per line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealBasicVSR 推理")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from videox.engines.realbasicvsr.infer import RealBasicRequest, run_realbasicvsr

    run_realbasicvsr(
        RealBasicRequest(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_dir=Path(args.model_dir),
            max_seq_len=args.max_seq_len,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", flush=True)
        raise SystemExit(1) from exc
