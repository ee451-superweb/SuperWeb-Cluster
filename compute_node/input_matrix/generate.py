#!/usr/bin/env python3
"""Generate the shared input matrix/vector dataset.

This script exists so higher-level tools such as `benchmark.py` can stay simple:

- if `A.bin` and `x.bin` already exist, benchmark runs immediately
- if they do not exist, benchmark shells out to this script once

By default the script generates the current production-sized dataset:

- `A`: 16384 x 32768 float32, exactly 2 GiB
- `x`: 32768 float32

`--rows` and `--cols` are kept so tests can generate a tiny dataset quickly.
The generated files themselves are generic binary assets, not tied to a single
compute backend implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_node.input_matrix import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    generate_dataset,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "generated"


def _to_relative_string(path: Path | str, *, start: Path) -> str:
    """Render a path relative to `start` without leaking machine-specific prefixes."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")
    return os.path.relpath(str(candidate), str(start)).replace("\\", "/")


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

    parser = argparse.ArgumentParser(description="Generate the shared matrix/vector input dataset.")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional generator worker count. Defaults to an automatic threaded setting for large datasets.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=32,
        help="Chunk size in MiB used while streaming the dataset to disk. Defaults to 32 MiB.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset unless a matching one is already present."""

    args = build_parser().parse_args(argv)
    spec = build_input_matrix_spec(rows=args.rows, cols=args.cols)
    layout = build_dataset_layout(args.output_dir)
    display_root = _to_relative_string(layout.root_dir, start=PROJECT_ROOT)

    if not args.force and dataset_is_generated(layout, spec):
        print(f"dataset already present at {display_root}", flush=True)
        return 0

    print(
        f"generating dataset at {display_root} "
        f"(rows={spec.rows}, cols={spec.cols}, matrix_bytes={spec.matrix_bytes}, "
        f"workers={args.workers or 'auto'}, chunk_mib={args.chunk_mib})",
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
        generate_dataset(
            layout,
            spec,
            progress=_report_progress,
            generator_workers=args.workers,
            chunk_values=max(1, (args.chunk_mib * 1024 * 1024) // 4),
        )
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
