#!/usr/bin/env python3
"""Generate the fixed benchmark dataset.

This script exists so `benchmark.py` can stay simple:

- if `A.bin` and `x.bin` already exist, benchmark runs immediately
- if they do not exist, benchmark shells out to this script once

By default the script generates the requested production-sized dataset:

- `A`: 16384 x 32768 float32, exactly 2 GiB
- `x`: 32768 float32

`--rows` and `--cols` are kept so tests can generate a tiny dataset quickly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parents[1] / "performance metrics"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

from fmvm_dataset import build_dataset_layout, dataset_is_generated, generate_dataset
from path_utils import to_relative_string
from workloads import build_benchmark_spec

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "generated"


def _format_binary_size(num_bytes: int) -> str:
    """Render a byte count in a human-friendly binary unit."""

    kib = 1024
    mib = kib * 1024
    gib = mib * 1024
    if num_bytes >= gib:
        return f"{num_bytes / gib:.3f} GiB"
    if num_bytes >= mib:
        return f"{num_bytes / mib:.3f} MiB"
    if num_bytes >= kib:
        return f"{num_bytes / kib:.3f} KiB"
    return f"{num_bytes} B"


def build_parser() -> argparse.ArgumentParser:
    """Describe the CLI for dataset generation."""

    parser = argparse.ArgumentParser(description="Generate fixed matrix-vector benchmark data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where A.bin, x.bin, and dataset_meta.json are written.",
    )
    parser.add_argument("--rows", type=int, help="Optional test-only override for the matrix row count.")
    parser.add_argument("--cols", type=int, help="Optional test-only override for the matrix column count.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the dataset even if matching files already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset unless a matching one is already present."""

    args = build_parser().parse_args(argv)
    spec = build_benchmark_spec(rows=args.rows, cols=args.cols)
    layout = build_dataset_layout(args.output_dir)
    display_root = to_relative_string(layout.root_dir, start=PERF_DIR)

    if not args.force and dataset_is_generated(layout, spec):
        print(f"dataset already present at {display_root}", flush=True)
        return 0

    print(
        f"generating dataset at {display_root} "
        f"(rows={spec.rows}, cols={spec.cols}, matrix_bytes={spec.matrix_bytes})",
        flush=True,
    )

    bars: dict[str, object] = {}

    def _report_progress(label: str, written_bytes: int, total_bytes: int) -> None:
        if tqdm is None:
            print(
                f"{label}: wrote {_format_binary_size(written_bytes)} / {_format_binary_size(total_bytes)}",
                flush=True,
            )
            return

        bar = bars.get(label)
        if bar is None:
            bar = tqdm(
                total=total_bytes,
                desc=label,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )
            bars[label] = bar

        delta = written_bytes - bar.n
        if delta > 0:
            bar.update(delta)
        if written_bytes >= total_bytes:
            bar.close()
            bars.pop(label, None)

    try:
        generate_dataset(layout, spec, progress=_report_progress)
    finally:
        for bar in list(bars.values()):
            bar.close()
    print(
        f"generated dataset at {display_root} "
        f"(rows={spec.rows}, cols={spec.cols}, matrix_bytes={spec.matrix_bytes})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
