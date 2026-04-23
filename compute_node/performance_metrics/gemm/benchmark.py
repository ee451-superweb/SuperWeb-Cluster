"""Run the GEMM benchmark and emit the method-local result report.

Use this module when the compute node needs to produce a ``result.json``
advertising its cuBLAS GEMM capacity to the main node. GEMM is cuBLAS-only,
so there is no backend sweep or autotune: the runner picks its own kernel
per (M, N, K, device). This module just measures one mid-size pass and
writes out the stable summary shape the performance_summary loader consumes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import METHOD_GEMM
from compute_node.compute_methods.gemm.paths import (
    CUDA_BUILD_DIR,
    CUDA_EXECUTABLE_PATH,
    CUDA_SOURCE_PATH,
    GEMM_METHOD_DIR,
)
from compute_node.input_matrix.gemm import (
    build_dataset_layout,
    build_spec,
    dataset_is_generated,
    dataset_prefix_for_size,
    get_mid_spec,
    normalize_size_variant,
)
from compute_node.performance_metrics.gemm.config import (
    DATASET_DIR,
    DISPLAY_NAME,
    GENERATE_SCRIPT_PATH,
    RESULT_PATH,
)

DEFAULT_ITERATION_COUNT = 5
BENCHMARK_WORKLOAD_SIZE = "mid"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the GEMM benchmark entrypoint."""
    parser = argparse.ArgumentParser(description="Benchmark cuBLAS GEMM.")
    parser.add_argument(
        "--size",
        default=BENCHMARK_WORKLOAD_SIZE,
        choices=("small", "mid", "large"),
        help="Workload variant to measure (default: mid).",
    )
    parser.add_argument(
        "--iteration-count",
        type=int,
        default=DEFAULT_ITERATION_COUNT,
        help="Number of measured cuBLAS calls (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_PATH,
        help="Where to write the method-local result.json.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory holding generated A/B matrices.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the cuBLAS runner even if the executable already exists.",
    )
    return parser


_GEMM_GENCODE_ARGS = (
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_87,code=sm_87",
    "-gencode=arch=compute_88,code=sm_88",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_100,code=sm_100",
    "-gencode=arch=compute_103,code=sm_103",
    "-gencode=arch=compute_110,code=sm_110",
    "-gencode=arch=compute_120,code=sm_120",
    "-gencode=arch=compute_121,code=sm_121",
    "-gencode=arch=compute_120,code=compute_120",
)


def _ensure_runner_built(force_rebuild: bool = False) -> Path:
    """Compile the cuBLAS runner when the expected executable is missing.

    Returns:
        The path to the freshly built (or already present) executable.
    """
    if CUDA_EXECUTABLE_PATH.exists() and not force_rebuild:
        return CUDA_EXECUTABLE_PATH
    CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        _compile_windows_runner()
    else:
        _compile_posix_runner()
    return CUDA_EXECUTABLE_PATH


def _compile_posix_runner() -> None:
    """Compile the cuBLAS runner on Linux/macOS where nvcc uses the host compiler directly."""
    command = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-lcublas",
        "-o",
        str(CUDA_EXECUTABLE_PATH),
        str(CUDA_SOURCE_PATH),
        *_GEMM_GENCODE_ARGS,
    ]
    subprocess.run(command, check=True, cwd=GEMM_METHOD_DIR)


def _compile_windows_runner() -> None:
    """Compile the cuBLAS runner on Windows through a VsDevCmd-wrapped batch script.

    Why: nvcc on Windows dispatches the host-side compile to ``cl.exe``. Unless
    the shell has already sourced VsDevCmd.bat, ``cl.exe`` is not on PATH and
    nvcc exits with ``Cannot find compiler 'cl.exe'``. conv2d's CudaBackend
    solves this by generating a .cmd wrapper that calls VsDevCmd first and then
    nvcc; we reuse its VsDevCmd discovery helper so both methods agree on which
    toolchain to use.
    """
    # Reuse conv2d's VsDevCmd discovery so both methods find the same toolchain.
    from compute_node.performance_metrics.conv2d.backends.cuda_backend import CudaBackend

    vsdevcmd = CudaBackend._find_vsdevcmd()
    if vsdevcmd is None:
        raise FileNotFoundError(
            "VsDevCmd.bat was not found; install Visual Studio Build Tools and rerun with --rebuild"
        )
    compile_script_path = CUDA_BUILD_DIR / "build_gemm_cuda_runner.cmd"
    nvcc_line = " ".join(
        [
            "nvcc",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "-Wno-deprecated-gpu-targets",
            "-cudart",
            "static",
            "-Xcompiler",
            "\"/MT /EHsc\"",
            "-o",
            CUDA_EXECUTABLE_PATH.name,
            f"..\\{CUDA_SOURCE_PATH.name}",
            "-lcublas",
            *_GEMM_GENCODE_ARGS,
        ]
    )
    compile_script_path.write_text(
        "\n".join(
            [
                "@echo off",
                f"call \"{vsdevcmd}\" -arch=x64 -host_arch=x64 >nul",
                "if errorlevel 1 exit /b %errorlevel%",
                "pushd \"%~dp0\"",
                nvcc_line,
                "set \"BUILD_EXIT=%ERRORLEVEL%\"",
                "popd",
                "exit /b %BUILD_EXIT%",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    completed = subprocess.run(
        ["cmd", "/c", str(compile_script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=CUDA_BUILD_DIR,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _ensure_dataset_ready(dataset_dir: Path, variant: str) -> None:
    """Generate the GEMM dataset when the requested variant is missing."""
    spec = build_spec(default_variant=variant)
    layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size(variant))
    if dataset_is_generated(layout, spec):
        return
    command = [
        sys.executable,
        str(GENERATE_SCRIPT_PATH),
        "--output-dir",
        str(dataset_dir),
    ]
    subprocess.run(command, check=True)


def _run_benchmark(variant: str, iteration_count: int, dataset_dir: Path) -> dict:
    """Run one cuBLAS benchmark pass and return the parsed runner record."""
    spec = build_spec(default_variant=variant)
    layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size(variant))
    output_path = dataset_dir / ".benchmark_C.bin"
    command = [
        str(CUDA_EXECUTABLE_PATH),
        "--input-a",
        str(layout.a_path),
        "--input-b",
        str(layout.b_path),
        "--output",
        str(output_path),
        "--m",
        str(spec.m),
        "--n",
        str(spec.n),
        "--k",
        str(spec.k),
        "--iteration-count",
        str(iteration_count),
        "--mode",
        "benchmark",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=GEMM_METHOD_DIR,
            timeout=1800.0,
        )
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
    return json.loads(completed.stdout)


def _build_result_payload(
    *,
    variant: str,
    iteration_count: int,
    record: dict,
    benchmark_elapsed_seconds: float,
) -> dict:
    """Assemble the result.json payload from one benchmark pass."""
    effective_gflops = float(record.get("effective_gflops") or 0.0)
    compute_event_ms = float(record.get("compute_event_ms") or 0.0)
    per_iter_seconds = float(record.get("per_iter_wall_clock_latency_seconds") or 0.0)
    spec = build_spec(default_variant=variant)
    return {
        "method": METHOD_GEMM,
        "display_name": DISPLAY_NAME,
        "generated_at_unix": time.time(),
        "benchmark_elapsed_seconds": benchmark_elapsed_seconds,
        "dataset": {
            "root_dir": str(DATASET_DIR),
            "variant": variant,
            "shape": {"m": spec.m, "n": spec.n, "k": spec.k},
        },
        "backends_considered": ["cuda"],
        "detected_backends": ["cuda"],
        "usable_backends": ["cuda"],
        "ranking": ["cuda"],
        "best_backend": "cuda",
        "backends": {
            "cuda": {
                "backend": "cuda",
                "available": True,
                "rank": 1,
                "autotune_plan": {
                    "autotune_repeats": 0,
                    "measurement_repeats": iteration_count,
                    "trials_run": 1,
                    "search_space": {},
                },
                "best_config": {
                    # cuBLAS picks its own kernel per (M, N, K, device); there is
                    # no knob to capture here. Kept as an empty mapping so the
                    # performance_summary loader accepts the entry.
                    "measurement_repeats": iteration_count,
                },
                "best_result": {
                    "wall_clock_latency_seconds": per_iter_seconds,
                    "effective_gflops": effective_gflops,
                    "compute_event_ms": compute_event_ms,
                    "checksum": str(record.get("checksum") or ""),
                },
                "notes": [
                    "cuBLAS SGEMM; no kernel sweep; M=N=K sized by workload.",
                ],
            }
        },
    }


def run(
    *,
    size: str = BENCHMARK_WORKLOAD_SIZE,
    iteration_count: int = DEFAULT_ITERATION_COUNT,
    output: Path | None = None,
    dataset_dir: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Run the GEMM benchmark and write the method-local result.json."""
    variant = normalize_size_variant(size, default="mid")
    target_output = RESULT_PATH if output is None else Path(output)
    target_dataset = DATASET_DIR if dataset_dir is None else Path(dataset_dir)

    _ensure_runner_built(force_rebuild=force_rebuild)
    _ensure_dataset_ready(target_dataset, variant)

    started_at = time.monotonic()
    record = _run_benchmark(variant, iteration_count, target_dataset)
    elapsed_seconds = max(0.0, time.monotonic() - started_at)

    payload = _build_result_payload(
        variant=variant,
        iteration_count=iteration_count,
        record=record,
        benchmark_elapsed_seconds=elapsed_seconds,
    )
    target_output.parent.mkdir(parents=True, exist_ok=True)
    target_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_output


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for standalone GEMM benchmark runs."""
    args = build_parser().parse_args(argv)
    path = run(
        size=args.size,
        iteration_count=args.iteration_count,
        output=args.output,
        dataset_dir=args.dataset_dir,
        force_rebuild=args.force_rebuild,
    )
    print(f"gemm benchmark wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
