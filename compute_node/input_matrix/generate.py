#!/usr/bin/env python3
"""Generate input datasets for one or more compute methods.

Use this module as the top-level dataset-generation CLI when the project should
prepare GEMV, conv2d, or both datasets in one invocation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode

enable_utf8_mode()

from core.venv import relaunch_with_project_python_if_needed
from core.constants import METHOD_GEMV, METHOD_CONV2D, METHOD_GEMM
from compute_node.input_matrix.gemv.generate import main as generate_gemv_main
from compute_node.input_matrix.conv2d.generate import main as generate_conv2d_main
from compute_node.input_matrix.gemm.generate import main as generate_gemm_main

DEFAULT_GENERATOR_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CHUNK_MIB = 8


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the top-level dataset generator.

    Returns:
        The configured ``ArgumentParser`` for dataset generation.
    """
    parser = argparse.ArgumentParser(description="Generate input datasets for one or more compute methods.")
    parser.add_argument(
        "--method",
        choices=(METHOD_GEMV, METHOD_CONV2D, METHOD_GEMM, "all"),
        default="all",
        help="Which method dataset to generate. Default: generate all in sequence.",
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
    parser.add_argument("--skip-small", action="store_true", help="Skip the small dataset.")
    parser.add_argument("--skip-mid", action="store_true", help="Skip the mid-sized dataset.")
    parser.add_argument("--skip-medium", action="store_true", help="Alias for --skip-mid.")
    parser.add_argument("--skip-large", action="store_true", help="Skip the large dataset.")
    parser.add_argument("--skip-test", action="store_true", help="Alias for --skip-small.")
    parser.add_argument("--skip-runtime", action="store_true", help="Alias for --skip-large.")
    parser.add_argument("--rows", type=int, help="GEMV test-only row-count override.")
    parser.add_argument("--cols", type=int, help="GEMV test-only column-count override.")
    parser.add_argument("--role", choices=("compute", "main"), default="compute", help="Conv2d large-dataset role.")
    parser.add_argument("--h", type=int, help="Conv2d small height override.")
    parser.add_argument("--w", type=int, help="Conv2d small width override.")
    parser.add_argument("--cin", type=int, help="Conv2d input-channel override.")
    parser.add_argument("--cout", type=int, help="Conv2d output-channel override.")
    parser.add_argument("--k", type=int, help="Conv2d kernel-size override.")
    parser.add_argument("--pad", type=int, help="Conv2d padding override.")
    parser.add_argument("--stride", type=int, help="Conv2d stride override.")
    parser.add_argument("--include-large-weight", action="store_true", help="Generate large weights for compute-role conv2d datasets.")
    parser.add_argument("--include-runtime-weight", action="store_true", help="Alias for --include-large-weight.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit extra per-method forwarded-argument details to stdout.",
    )
    return parser


def _selected_methods(method_arg: str) -> list[str]:
    """Expand the CLI method selector into an ordered method list.

    Args:
        method_arg: CLI ``--method`` value.

    Returns:
        The ordered list of methods whose datasets should be generated.
    """
    if method_arg == "all":
        # Match the benchmark ordering (GEMM first) so dataset generation
        # finishes its fastest method first and is immediately ready when the
        # benchmark picks up the same ordering.
        return [
            METHOD_GEMM,
            METHOD_GEMV,
            METHOD_CONV2D,
        ]
    return [method_arg]


def _common_flags(args: argparse.Namespace) -> list[str]:
    """Build the shared CLI flags forwarded to method-local generators.

    Args:
        args: Parsed top-level dataset-generation CLI arguments.

    Returns:
        The list of shared flags to forward to method-local generators.
    """
    argv: list[str] = []
    if args.force:
        argv.append("--force")
    if args.workers is not None:
        argv.extend(["--workers", str(args.workers)])
    if args.chunk_mib is not None:
        argv.extend(["--chunk-mib", str(args.chunk_mib)])
    if args.skip_small or args.skip_test:
        argv.append("--skip-small")
    if args.skip_mid or args.skip_medium:
        argv.append("--skip-mid")
    if args.skip_large or args.skip_runtime:
        argv.append("--skip-large")
    return argv


def _build_gemv_args(args: argparse.Namespace) -> list[str]:
    """Build the forwarded CLI argument list for the GEMV generator.

    Args:
        args: Parsed top-level dataset-generation CLI arguments.

    Returns:
        The GEMV-specific forwarded CLI argument list.
    """
    argv = _common_flags(args)
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if args.rows is not None:
        argv.extend(["--rows", str(args.rows)])
    if args.cols is not None:
        argv.extend(["--cols", str(args.cols)])
    return argv


def _build_gemm_args(args: argparse.Namespace) -> list[str]:
    """Build the forwarded CLI argument list for the GEMM generator."""
    argv = _common_flags(args)
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    return argv


def _build_conv2d_args(args: argparse.Namespace) -> list[str]:
    """Build the forwarded CLI argument list for the conv2d generator.

    Args:
        args: Parsed top-level dataset-generation CLI arguments.

    Returns:
        The conv2d-specific forwarded CLI argument list.
    """
    argv = _common_flags(args)
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    argv.extend(["--role", str(args.role)])
    if args.include_large_weight or args.include_runtime_weight:
        argv.append("--include-large-weight")
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
    """CLI entrypoint for unified input-dataset generation.

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
    methods = _selected_methods(args.method)
    if args.output_dir is not None and len(methods) > 1:
        raise SystemExit("--output-dir can only be used when generating one method at a time")

    verbose = bool(getattr(args, "verbose", False))
    for method_name in methods:
        if method_name == METHOD_GEMV:
            forwarded = _build_gemv_args(args)
            if verbose:
                print(f"[input_matrix verbose] gemv forwarded_args={forwarded}", flush=True)
            generate_gemv_main(forwarded)
        elif method_name == METHOD_CONV2D:
            forwarded = _build_conv2d_args(args)
            if verbose:
                print(f"[input_matrix verbose] conv2d forwarded_args={forwarded}", flush=True)
            generate_conv2d_main(forwarded)
        elif method_name == METHOD_GEMM:
            forwarded = _build_gemm_args(args)
            if verbose:
                print(f"[input_matrix verbose] gemm forwarded_args={forwarded}", flush=True)
            generate_gemm_main(forwarded)
        else:
            raise ValueError(f"unsupported input-matrix method: {method_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
