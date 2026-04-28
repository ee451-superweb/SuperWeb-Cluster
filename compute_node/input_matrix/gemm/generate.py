#!/usr/bin/env python3
"""Generate GEMM small, medium, and large datasets in the method-local workspace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode

enable_utf8_mode()

from core.venv import relaunch_with_project_python_if_needed
from compute_node.performance_metrics.gemm.config import DATASET_DIR
from compute_node.input_matrix.gemm import (
    build_dataset_layout,
    dataset_is_generated,
    dataset_prefix_for_size,
    generate_dataset,
    get_large_spec,
    get_mid_spec,
    get_small_spec,
)
from compute_node.input_matrix.progress import build_progress_reporter, emit_progress_message

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def _to_relative_string(path: Path | str, *, start: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")
    return os.path.relpath(str(candidate), str(start)).replace("\\", "/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GEMM small/mid/large datasets.")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
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
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--skip-mid", action="store_true")
    parser.add_argument("--skip-medium", action="store_true", help="Alias for --skip-mid.")
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Alias for --skip-small.")
    parser.add_argument("--skip-runtime", action="store_true", help="Alias for --skip-large.")
    return parser


def _maybe_generate(layout, spec, *, label, display_root, force, report_progress, workers, chunk_mib):
    if force or not dataset_is_generated(layout, spec):
        emit_progress_message(
            f"generating GEMM {label} dataset at {display_root} "
            f"(M={spec.m}, N={spec.n}, K={spec.k}, "
            f"A_bytes={spec.a_bytes}, B_bytes={spec.b_bytes}, "
            f"workers={workers}, chunk_mib={chunk_mib})"
        )
        generate_dataset(
            layout,
            spec,
            progress=report_progress,
            generator_workers=workers,
            chunk_values=max(1, (chunk_mib * 1024 * 1024) // 4),
        )
    else:
        emit_progress_message(f"GEMM {label} dataset already present at {display_root}")


def main(argv: list[str] | None = None) -> int:
    relaunch_result = relaunch_with_project_python_if_needed(
        argv,
        script_path=Path(__file__),
        cwd=PROJECT_ROOT,
    )
    if relaunch_result is not None:
        return relaunch_result

    args = build_parser().parse_args(argv)
    skip_small = bool(args.skip_small or args.skip_test)
    skip_mid = bool(args.skip_mid or args.skip_medium)
    skip_large = bool(args.skip_large or args.skip_runtime)
    if skip_small and skip_mid and skip_large:
        raise ValueError("cannot skip small, mid, and large dataset generation at the same time")

    display_root = _to_relative_string(args.output_dir, start=PROJECT_ROOT)
    report_progress, close_progress = build_progress_reporter()
    try:
        if not skip_small:
            small_spec = get_small_spec()
            small_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("small"))
            _maybe_generate(
                small_layout, small_spec,
                label="small", display_root=display_root, force=args.force,
                report_progress=report_progress, workers=args.workers, chunk_mib=args.chunk_mib,
            )
        if not skip_mid:
            mid_spec = get_mid_spec()
            mid_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("mid"))
            _maybe_generate(
                mid_layout, mid_spec,
                label="mid", display_root=display_root, force=args.force,
                report_progress=report_progress, workers=args.workers, chunk_mib=args.chunk_mib,
            )
        if not skip_large:
            large_spec = get_large_spec()
            large_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("large"))
            _maybe_generate(
                large_layout, large_spec,
                label="large", display_root=display_root, force=args.force,
                report_progress=report_progress, workers=args.workers, chunk_mib=args.chunk_mib,
            )
    finally:
        close_progress()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
