#!/usr/bin/env python3
"""Unified input-dataset generation entrypoint for all compute methods."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime_environment import relaunch_with_project_python_if_needed
from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from compute_node.input_matrix.fixed_matrix_vector_multiplication.generate import main as generate_fmvm_main
from compute_node.input_matrix.spatial_convolution.generate import main as generate_spatial_main

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate input datasets for one or more compute methods.")
    parser.add_argument(
        "--method",
        choices=(METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION, "all"),
        default="all",
        help="Which method dataset to generate. Default: generate both in sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the generated dataset directory for a single selected method.",
    )
    parser.add_argument("--force", action="store_true", help="Rewrite matching datasets.")
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
    parser.add_argument("--rows", type=int, help="FMVM test-only row-count override.")
    parser.add_argument("--cols", type=int, help="FMVM test-only column-count override.")
    parser.add_argument("--role", choices=("compute", "main"), default="compute", help="Spatial-convolution runtime dataset role.")
    parser.add_argument("--h", type=int, help="Spatial-convolution test height override.")
    parser.add_argument("--w", type=int, help="Spatial-convolution test width override.")
    parser.add_argument("--cin", type=int, help="Spatial-convolution input-channel override.")
    parser.add_argument("--cout", type=int, help="Spatial-convolution output-channel override.")
    parser.add_argument("--k", type=int, help="Spatial-convolution kernel-size override.")
    parser.add_argument("--pad", type=int, help="Spatial-convolution padding override.")
    parser.add_argument("--stride", type=int, help="Spatial-convolution stride override.")
    parser.add_argument("--include-runtime-weight", action="store_true", help="Generate runtime weights for compute-role spatial-convolution datasets.")
    return parser


def _selected_methods(method_arg: str) -> list[str]:
    if method_arg == "all":
        return [
            METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
            METHOD_SPATIAL_CONVOLUTION,
        ]
    return [method_arg]


def _common_flags(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.force:
        argv.append("--force")
    if args.workers is not None:
        argv.extend(["--workers", str(args.workers)])
    if args.chunk_mib is not None:
        argv.extend(["--chunk-mib", str(args.chunk_mib)])
    if args.skip_runtime:
        argv.append("--skip-runtime")
    return argv


def _build_fmvm_args(args: argparse.Namespace) -> list[str]:
    argv = _common_flags(args)
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if args.rows is not None:
        argv.extend(["--rows", str(args.rows)])
    if args.cols is not None:
        argv.extend(["--cols", str(args.cols)])
    return argv


def _build_spatial_args(args: argparse.Namespace) -> list[str]:
    argv = _common_flags(args)
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    argv.extend(["--role", str(args.role)])
    if args.include_runtime_weight:
        argv.append("--include-runtime-weight")
    for field, cli_flag in (
        ("h", "--h"),
        ("w", "--w"),
        ("cin", "--cin"),
        ("cout", "--cout"),
        ("k", "--k"),
        ("pad", "--pad"),
        ("stride", "--stride"),
    ):
        value = getattr(args, field)
        if value is not None:
            argv.extend([cli_flag, str(value)])
    return argv


def main(argv: list[str] | None = None) -> int:
    relaunch_result = relaunch_with_project_python_if_needed(
        argv,
        script_path=Path(__file__),
        cwd=PROJECT_ROOT,
    )
    if relaunch_result is not None:
        return relaunch_result

    args = build_parser().parse_args(argv)
    methods = _selected_methods(args.method)
    if args.output_dir is not None and len(methods) > 1:
        raise SystemExit("--output-dir can only be used when generating one method at a time")

    for method_name in methods:
        if method_name == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
            generate_fmvm_main(_build_fmvm_args(args))
        elif method_name == METHOD_SPATIAL_CONVOLUTION:
            generate_spatial_main(_build_spatial_args(args))
        else:
            raise ValueError(f"unsupported input-matrix method: {method_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
