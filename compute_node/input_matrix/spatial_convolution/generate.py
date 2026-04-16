#!/usr/bin/env python3
"""Generate spatial-convolution test/runtime datasets in the method-local workspace."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime_environment import relaunch_with_project_python_if_needed
from compute_node.performance_metrics.spatial_convolution.config import DATASET_DIR
from compute_node.input_matrix.spatial_convolution import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    generate_dataset,
    get_runtime_input_matrix_spec,
)
from compute_node.input_matrix.progress import build_progress_reporter

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def _benchmark_fields_match(layout, spec) -> bool:
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
    parser = argparse.ArgumentParser(description="Generate spatial-convolution test/runtime datasets.")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--role", choices=["compute", "main"], default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--cin", type=int)
    parser.add_argument("--cout", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pad", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--include-runtime-weight", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
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
    relaunch_result = relaunch_with_project_python_if_needed(
        argv,
        script_path=Path(__file__),
        cwd=PROJECT_ROOT,
    )
    if relaunch_result is not None:
        return relaunch_result

    args = build_parser().parse_args(argv)
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
            default_variant="test",
        )
        if custom_requested
        else build_input_matrix_spec(default_variant="test")
    )
    runtime_spec = get_runtime_input_matrix_spec()
    chunk_values = max(1, (args.chunk_mib * 1024 * 1024) // 4)

    try:
        test_layout = build_dataset_layout(args.output_dir, prefix="test_")
        if args.force:
            _reset_stale_files(test_layout, test_spec, skip_weight=False)
        elif not dataset_is_generated(test_layout, test_spec, skip_weight=False):
            _reset_stale_files(test_layout, test_spec, skip_weight=False)
        if args.force or not dataset_is_generated(test_layout, test_spec, skip_weight=False):
            print(f"\n[Data Gen] Creating spatial-convolution TEST dataset ({test_spec.h}x{test_spec.w})...")
            generate_dataset(
                test_layout,
                test_spec,
                skip_weight=False,
                progress=_progress,
                generator_workers=args.workers,
                chunk_values=chunk_values,
            )

        if not args.skip_runtime:
            runtime_layout = build_dataset_layout(args.output_dir, prefix="runtime_")
            skip_rt_weight = (args.role == "compute") and not args.include_runtime_weight
            if args.force:
                _reset_stale_files(runtime_layout, runtime_spec, skip_weight=skip_rt_weight)
            elif not dataset_is_generated(runtime_layout, runtime_spec, skip_weight=skip_rt_weight):
                _reset_stale_files(runtime_layout, runtime_spec, skip_weight=skip_rt_weight)
            if args.force or not dataset_is_generated(runtime_layout, runtime_spec, skip_weight=skip_rt_weight):
                needs_input = not runtime_layout.input_path.exists()
                needs_weight = not skip_rt_weight and not runtime_layout.weight_path.exists()
                label = "RUNTIME INPUT" if needs_input and skip_rt_weight and not needs_weight else "FULL RUNTIME"
                print(f"\n[Data Gen] Creating spatial-convolution {label} ({runtime_spec.h}x{runtime_spec.w})...")
                generate_dataset(
                    runtime_layout,
                    runtime_spec,
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
