#!/usr/bin/env python3
"""Generate GEMV small, medium, and large datasets in the method-local workspace.

Use this module when the project needs GEMV benchmark or runtime inputs to be
generated on disk.
"""

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
from compute_node.performance_metrics.gemv.config import DATASET_DIR
from compute_node.input_matrix.gemv import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_prefix_for_size,
    dataset_is_generated,
    generate_dataset,
    get_large_input_matrix_spec,
    get_mid_input_matrix_spec,
)
from compute_node.input_matrix.progress import build_progress_reporter, emit_progress_message

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def _to_relative_string(path: Path | str, *, start: Path) -> str:
    """Render one path relative to a chosen base directory.

    Args:
        path: Path that should be displayed relative to ``start``.
        start: Base directory used for the relative display.

    Returns:
        A slash-normalized relative path string.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")
    return os.path.relpath(str(candidate), str(start)).replace("\\", "/")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the GEMV dataset generator.

    Returns:
        The configured ``ArgumentParser`` for GEMV dataset generation.
    """
    parser = argparse.ArgumentParser(description="Generate GEMV small/mid/large datasets.")
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
    parser.add_argument("--skip-small", action="store_true", help="Skip the small dataset.")
    parser.add_argument("--skip-mid", action="store_true", help="Skip the mid-sized dataset.")
    parser.add_argument("--skip-medium", action="store_true", help="Alias for --skip-mid.")
    parser.add_argument("--skip-large", action="store_true", help="Skip the large dataset.")
    parser.add_argument("--skip-test", action="store_true", help="Alias for --skip-small.")
    parser.add_argument("--skip-runtime", action="store_true", help="Alias for --skip-large.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for GEMV dataset generation.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code ``0`` on success.
    """
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

    small_spec = build_input_matrix_spec(rows=args.rows, cols=args.cols, default_variant="small")
    mid_spec = get_mid_input_matrix_spec()
    large_spec = get_large_input_matrix_spec()
    display_root = _to_relative_string(args.output_dir, start=PROJECT_ROOT)
    chunk_values = max(1, (args.chunk_mib * 1024 * 1024) // 4)
    _report_progress, close_progress = build_progress_reporter()

    try:
        if not skip_small:
            small_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("small"))
            if args.force or not dataset_is_generated(small_layout, small_spec):
                emit_progress_message(
                    f"generating GEMV small dataset at {display_root} "
                    f"(rows={small_spec.rows}, cols={small_spec.cols}, matrix_bytes={small_spec.matrix_bytes}, "
                    f"workers={args.workers}, chunk_mib={args.chunk_mib})"
                )
                generate_dataset(
                    small_layout,
                    small_spec,
                    progress=_report_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
            else:
                emit_progress_message(f"GEMV small dataset already present at {display_root}")

        if not skip_mid:
            mid_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("mid"))
            if args.force or not dataset_is_generated(mid_layout, mid_spec):
                emit_progress_message(
                    f"generating GEMV mid dataset at {display_root} "
                    f"(rows={mid_spec.rows}, cols={mid_spec.cols}, matrix_bytes={mid_spec.matrix_bytes}, "
                    f"workers={args.workers}, chunk_mib={args.chunk_mib})"
                )
                generate_dataset(
                    mid_layout,
                    mid_spec,
                    progress=_report_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
            else:
                emit_progress_message(f"GEMV mid dataset already present at {display_root}")

        if not skip_large:
            large_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("large"))
            if args.force or not dataset_is_generated(large_layout, large_spec):
                emit_progress_message(
                    f"generating GEMV large dataset at {display_root} "
                    f"(rows={large_spec.rows}, cols={large_spec.cols}, matrix_bytes={large_spec.matrix_bytes}, "
                    f"workers={args.workers}, chunk_mib={args.chunk_mib})"
                )
                generate_dataset(
                    large_layout,
                    large_spec,
                    progress=_report_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
            else:
                emit_progress_message(f"GEMV large dataset already present at {display_root}")
    finally:
        close_progress()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
