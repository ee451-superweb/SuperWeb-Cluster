#!/usr/bin/env python3
"""Generate FMVM test/runtime datasets in the method-local workspace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime_environment import relaunch_with_project_python_if_needed
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import DATASET_DIR
from compute_node.input_matrix.fixed_matrix_vector_multiplication import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    generate_dataset,
    get_runtime_input_matrix_spec,
)
from compute_node.input_matrix.progress import build_progress_reporter

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def _to_relative_string(path: Path | str, *, start: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")
    return os.path.relpath(str(candidate), str(start)).replace("\\", "/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FMVM test/runtime datasets.")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--rows", type=int, help="Optional test-only row-count override.")
    parser.add_argument("--cols", type=int, help="Optional test-only column-count override.")
    parser.add_argument("--force", action="store_true", help="Rewrite datasets even if matching files already exist.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_GENERATOR_WORKERS,
        help="Generator worker count. Default: OS logical processor count.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=DEFAULT_CHUNK_MIB,
        help="Chunk size in MiB used while streaming data.",
    )
    parser.add_argument("--skip-runtime", action="store_true", help="Generate only the test dataset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    relaunch_result = relaunch_with_project_python_if_needed(
        argv,
        script_path=Path(__file__),
        cwd=PROJECT_ROOT,
    )
    if relaunch_result is not None:
        return relaunch_result

    args = build_parser().parse_args(argv)
    test_spec = build_input_matrix_spec(rows=args.rows, cols=args.cols, default_variant="test")
    runtime_spec = get_runtime_input_matrix_spec()
    display_root = _to_relative_string(args.output_dir, start=PROJECT_ROOT)
    chunk_values = max(1, (args.chunk_mib * 1024 * 1024) // 4)
    _report_progress, close_progress = build_progress_reporter()

    try:
        test_layout = build_dataset_layout(args.output_dir, prefix="test_")
        if args.force or not dataset_is_generated(test_layout, test_spec):
            print(
                f"generating FMVM test dataset at {display_root} "
                f"(rows={test_spec.rows}, cols={test_spec.cols}, matrix_bytes={test_spec.matrix_bytes}, "
                f"workers={args.workers}, chunk_mib={args.chunk_mib})",
                flush=True,
            )
            generate_dataset(
                test_layout,
                test_spec,
                progress=_report_progress,
                generator_workers=args.workers,
                chunk_values=chunk_values,
            )
        else:
            print(f"FMVM test dataset already present at {display_root}", flush=True)

        if not args.skip_runtime:
            runtime_layout = build_dataset_layout(args.output_dir, prefix="runtime_")
            if args.force or not dataset_is_generated(runtime_layout, runtime_spec):
                print(
                    f"generating FMVM runtime dataset at {display_root} "
                    f"(rows={runtime_spec.rows}, cols={runtime_spec.cols}, matrix_bytes={runtime_spec.matrix_bytes}, "
                    f"workers={args.workers}, chunk_mib={args.chunk_mib})",
                    flush=True,
                )
                generate_dataset(
                    runtime_layout,
                    runtime_spec,
                    progress=_report_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
            else:
                print(f"FMVM runtime dataset already present at {display_root}", flush=True)
    finally:
        close_progress()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
