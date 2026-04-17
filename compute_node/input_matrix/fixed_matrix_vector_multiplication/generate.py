#!/usr/bin/env python3
"""Generate FMVM small, medium, and large datasets in the method-local workspace.

Use this module when the project needs FMVM benchmark or runtime inputs to be
generated or refreshed on disk.
"""

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
    get_medium_input_matrix_spec,
    get_runtime_input_matrix_spec,
)
from compute_node.input_matrix.progress import build_progress_reporter

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
    """Build the CLI parser for the FMVM dataset generator.

    Returns:
        The configured ``ArgumentParser`` for FMVM dataset generation.
    """
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
    parser.add_argument("--skip-small", action="store_true", help="Skip the small dataset.")
    parser.add_argument("--skip-medium", action="store_true", help="Skip the medium dataset.")
    parser.add_argument("--skip-large", action="store_true", help="Skip the large dataset.")
    parser.add_argument("--skip-test", action="store_true", help="Alias for --skip-small.")
    parser.add_argument("--skip-runtime", action="store_true", help="Alias for --skip-large.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for FMVM dataset generation.

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
    skip_medium = bool(args.skip_medium)
    skip_large = bool(args.skip_large or args.skip_runtime)
    if skip_small and skip_medium and skip_large:
        raise ValueError("cannot skip small, medium, and large dataset generation at the same time")

    test_spec = build_input_matrix_spec(rows=args.rows, cols=args.cols, default_variant="test")
    medium_spec = get_medium_input_matrix_spec()
    runtime_spec = get_runtime_input_matrix_spec()
    display_root = _to_relative_string(args.output_dir, start=PROJECT_ROOT)
    chunk_values = max(1, (args.chunk_mib * 1024 * 1024) // 4)
    _report_progress, close_progress = build_progress_reporter()

    try:
        if not skip_small:
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

        if not skip_medium:
            medium_layout = build_dataset_layout(args.output_dir, prefix="medium_")
            if args.force or not dataset_is_generated(medium_layout, medium_spec):
                print(
                    f"generating FMVM medium dataset at {display_root} "
                    f"(rows={medium_spec.rows}, cols={medium_spec.cols}, matrix_bytes={medium_spec.matrix_bytes}, "
                    f"workers={args.workers}, chunk_mib={args.chunk_mib})",
                    flush=True,
                )
                generate_dataset(
                    medium_layout,
                    medium_spec,
                    progress=_report_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
            else:
                print(f"FMVM medium dataset already present at {display_root}", flush=True)

        if not skip_large:
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
