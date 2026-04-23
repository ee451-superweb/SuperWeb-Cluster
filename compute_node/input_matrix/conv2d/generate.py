#!/usr/bin/env python3
"""Generate conv2d small, medium, and large datasets.

Use this module when the project needs conv2d benchmark or runtime inputs to
be generated on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode

enable_utf8_mode()

from core.venv import relaunch_with_project_python_if_needed
from compute_node.performance_metrics.conv2d.config import DATASET_DIR
from compute_node.input_matrix.conv2d import (
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


def _benchmark_fields_match(layout, spec) -> bool:
    """Check whether dataset metadata still matches the requested benchmark spec.

    Args:
        layout: Dataset layout whose metadata should be inspected.
        spec: Benchmark spec that the dataset is expected to match.

    Returns:
        ``True`` when metadata matches the requested benchmark fields.
    """
    if not layout.meta_path.exists():
        return False

    try:
        metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    benchmark = metadata.get("benchmark", {})
    return (
        benchmark.get("h") == spec.h
        and benchmark.get("w") == spec.w
        and benchmark.get("c_in") == spec.c_in
        and benchmark.get("c_out") == spec.c_out
        and benchmark.get("k") == spec.k
        and benchmark.get("pad") == spec.pad
        and benchmark.get("stride", 1) == spec.stride
    )


def _reset_stale_files(layout, spec, *, skip_weight: bool) -> None:
    """Delete stale dataset files whose shape no longer matches the spec.

    Args:
        layout: Dataset layout that may contain stale files.
        spec: Benchmark spec that should define the dataset shape.
        skip_weight: Whether the weight file may be absent intentionally.

    Returns:
        ``None`` after stale files have been removed.
    """
    input_stale = layout.input_path.exists() and layout.input_path.stat().st_size != spec.input_bytes
    weight_stale = (
        not skip_weight
        and layout.weight_path.exists()
        and layout.weight_path.stat().st_size != spec.weight_bytes
    )
    meta_stale = layout.meta_path.exists() and not _benchmark_fields_match(layout, spec)

    if not (input_stale or weight_stale or meta_stale):
        return

    layout.input_path.unlink(missing_ok=True)
    layout.meta_path.unlink(missing_ok=True)
    if not skip_weight:
        layout.weight_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the conv2d dataset generator.

    Returns:
        The configured ``ArgumentParser`` for conv2d dataset generation.
    """
    parser = argparse.ArgumentParser(description="Generate conv2d small/mid/large datasets.")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--role", choices=["compute", "main"], default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--cin", type=int)
    parser.add_argument("--cout", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pad", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--include-large-weight", action="store_true")
    parser.add_argument("--include-runtime-weight", action="store_true", help="Alias for --include-large-weight.")
    parser.add_argument("--skip-small", action="store_true", help="Skip the small dataset.")
    parser.add_argument("--skip-mid", action="store_true", help="Skip the mid-sized dataset.")
    parser.add_argument("--skip-medium", action="store_true", help="Alias for --skip-mid.")
    parser.add_argument("--skip-large", action="store_true", help="Skip the large dataset.")
    parser.add_argument("--skip-test", action="store_true", help="Alias for --skip-small.")
    parser.add_argument("--skip-runtime", action="store_true", help="Alias for --skip-large.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for conv2d dataset generation.

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
        raise ValueError("cannot skip small, mid, and large dataset generation")

    _progress, close_progress = build_progress_reporter()

    custom_requested = any(
        value is not None for value in (args.h, args.w, args.cin, args.cout, args.k, args.pad, args.stride)
    )
    test_spec = (
        build_input_matrix_spec(
            h=args.h,
            w=args.w,
            c_in=args.cin,
            c_out=args.cout,
            k=args.k,
            pad=args.pad,
            stride=args.stride,
            default_variant="small",
        )
        if custom_requested
        else build_input_matrix_spec(default_variant="small")
    )
    mid_spec = get_mid_input_matrix_spec()
    large_spec = get_large_input_matrix_spec()
    include_large_weight = bool(args.include_large_weight or args.include_runtime_weight)
    chunk_values = max(1, (args.chunk_mib * 1024 * 1024) // 4)

    try:
        if not skip_small:
            small_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("small"))
            if args.force:
                _reset_stale_files(small_layout, test_spec, skip_weight=False)
            elif not dataset_is_generated(small_layout, test_spec, skip_weight=False):
                _reset_stale_files(small_layout, test_spec, skip_weight=False)
            if args.force or not dataset_is_generated(small_layout, test_spec, skip_weight=False):
                emit_progress_message(f"[Data Gen] Creating conv2d SMALL dataset ({test_spec.h}x{test_spec.w})...")
                generate_dataset(
                    small_layout,
                    test_spec,
                    skip_weight=False,
                    progress=_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )

        if not skip_mid:
            mid_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("mid"))
            if args.force:
                _reset_stale_files(mid_layout, mid_spec, skip_weight=False)
            elif not dataset_is_generated(mid_layout, mid_spec, skip_weight=False):
                _reset_stale_files(mid_layout, mid_spec, skip_weight=False)
            if args.force or not dataset_is_generated(mid_layout, mid_spec, skip_weight=False):
                emit_progress_message(f"[Data Gen] Creating conv2d MID dataset ({mid_spec.h}x{mid_spec.w})...")
                generate_dataset(
                    mid_layout,
                    mid_spec,
                    skip_weight=False,
                    progress=_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )

        if not skip_large:
            large_layout = build_dataset_layout(args.output_dir, prefix=dataset_prefix_for_size("large"))
            skip_rt_weight = (args.role == "compute") and not include_large_weight
            if args.force:
                _reset_stale_files(large_layout, large_spec, skip_weight=skip_rt_weight)
            elif not dataset_is_generated(large_layout, large_spec, skip_weight=skip_rt_weight):
                _reset_stale_files(large_layout, large_spec, skip_weight=skip_rt_weight)
            if args.force or not dataset_is_generated(large_layout, large_spec, skip_weight=skip_rt_weight):
                needs_input = not large_layout.input_path.exists()
                needs_weight = not skip_rt_weight and not large_layout.weight_path.exists()
                label = "LARGE INPUT" if needs_input and skip_rt_weight and not needs_weight else "FULL LARGE"
                emit_progress_message(f"[Data Gen] Creating conv2d {label} ({large_spec.h}x{large_spec.w})...")
                generate_dataset(
                    large_layout,
                    large_spec,
                    skip_weight=skip_rt_weight,
                    progress=_progress,
                    generator_workers=args.workers,
                    chunk_values=chunk_values,
                )
        return 0
    finally:
        close_progress()


if __name__ == "__main__":
    raise SystemExit(main())
