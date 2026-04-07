#!/usr/bin/env python3
"""Generate deterministic fixed-matrix-vector benchmark inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fmvm_dataset import ensure_dataset
from workloads import PRESET_WORKLOADS, resolve_workload

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "generated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed matrix-vector benchmark data.")
    parser.add_argument(
        "--preset",
        default="standard",
        choices=sorted(PRESET_WORKLOADS),
        help="Named workload preset.",
    )
    parser.add_argument("--rows", type=int, help="Override workload rows.")
    parser.add_argument("--cols", type=int, help="Override workload cols.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where generated datasets are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload = resolve_workload(args.preset, rows=args.rows, cols=args.cols)
    dataset = ensure_dataset(args.output_dir, workload)
    print(f"generated dataset at {dataset.root_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
