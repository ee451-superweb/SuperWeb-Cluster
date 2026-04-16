"""FMVM benchmark implementation used by the top-level multi-method runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_node.input_matrix.fixed_matrix_vector_multiplication import build_dataset_layout
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import (
    DATASET_DIR as METHOD_DATASET_DIR,
    GENERATE_SCRIPT_PATH as METHOD_GENERATE_SCRIPT_PATH,
    RESULT_PATH as METHOD_RESULT_PATH,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends import build_backends
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.dataset_runner import (
    generate_dataset_if_missing,
    resolve_dataset_dir,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.reporting import (
    build_report,
    probe_backends,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.workloads import build_benchmark_spec
from compute_node.performance_metrics.path_utils import to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = METHOD_DATASET_DIR
DEFAULT_OUTPUT_PATH = METHOD_RESULT_PATH
GENERATE_SCRIPT_PATH = METHOD_GENERATE_SCRIPT_PATH


def build_parser() -> argparse.ArgumentParser:
    """Describe the CLI surface for the FMVM benchmark runner."""

    parser = argparse.ArgumentParser(description="Benchmark fixed matrix-vector multiplication backends.")
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend to run. Repeatable. Default: current auto-detected backend order.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where A.bin, x.bin, and dataset_meta.json live.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file that receives the benchmark report.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=240.0,
        help="Total benchmark time budget in seconds. Defaults to 240 (< 5 minutes).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force backend executables to rebuild instead of reusing checked-in or cached binaries.",
    )
    parser.add_argument(
        "--accumulation-precision",
        choices=("fp32", "fp64_accumulate"),
        default="fp32",
        help="Numeric accumulation mode. Default: fp32. fp64_accumulate is slower but can reduce tiny cross-backend drift.",
    )
    parser.add_argument("--rows", type=int, help="Optional test-only override for the matrix row count.")
    parser.add_argument("--cols", type=int, help="Optional test-only override for the matrix column count.")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the FMVM benchmark and return the JSON-serializable report."""

    emit_status(
        "method.fmvm.start",
        status="running",
        method="fixed_matrix_vector_multiplication",
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        rows=args.rows,
        cols=args.cols,
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
    )
    backends = build_backends(args.backend)
    force_rebuild = bool(getattr(args, "rebuild", False))
    hardware_inventory = probe_backends(backends)
    detected_backends = [
        backend.name
        for backend in backends
        if bool((hardware_inventory.get(backend.name) or {}).get("probe_available"))
    ]

    accumulation_precision = getattr(args, "accumulation_precision", "fp32")
    spec = build_benchmark_spec(
        default_variant="test",
        rows=args.rows,
        cols=args.cols,
        accumulation_precision=accumulation_precision,
    )
    use_runtime_measurement = args.rows is None and args.cols is None
    measurement_spec = (
        build_benchmark_spec(
            default_variant="runtime",
            accumulation_precision=accumulation_precision,
        )
        if use_runtime_measurement
        else spec
    )
    dataset_dir = resolve_dataset_dir(args, spec, default_dataset_dir=DEFAULT_DATASET_DIR)
    autotune_dataset = build_dataset_layout(dataset_dir, prefix="test_")
    measurement_dataset = (
        build_dataset_layout(dataset_dir, prefix="runtime_")
        if use_runtime_measurement
        else autotune_dataset
    )
    dataset_was_generated = False
    if detected_backends:
        emit_status(
            "method.fmvm.dataset.check",
            status="running",
            method="fixed_matrix_vector_multiplication",
            dataset_dir=str(dataset_dir),
            require_runtime_measurement=use_runtime_measurement,
        )
        dataset_was_generated = generate_dataset_if_missing(
            dataset_dir,
            spec.rows,
            spec.cols,
            require_runtime_measurement=use_runtime_measurement,
            root_dir=ROOT_DIR,
            generate_script_path=GENERATE_SCRIPT_PATH,
        )

    total_started = time.perf_counter()
    backend_results = []
    runnable_backends = [backend for backend in backends if backend.name in detected_backends]
    runnable_index = 0
    for backend in backends:
        if backend.name not in detected_backends:
            emit_status(
                "method.fmvm.backend.skipped",
                status="running",
                method="fixed_matrix_vector_multiplication",
                backend=backend.name,
                reason="probe_unavailable",
                probe_message=(hardware_inventory.get(backend.name) or {}).get("probe_message"),
            )
            backend_results.append(
                backend.run(
                    spec,
                    autotune_dataset,
                    measurement_spec=measurement_spec,
                    measurement_dataset=measurement_dataset,
                    time_budget_seconds=1.0,
                    force_rebuild=force_rebuild,
                )
            )
            continue

        elapsed = time.perf_counter() - total_started
        remaining = max(args.time_budget - elapsed, 1.0)
        per_backend_budget = remaining / max(len(runnable_backends) - runnable_index, 1)
        emit_status(
            "method.fmvm.backend.start",
            status="running",
            method="fixed_matrix_vector_multiplication",
            backend=backend.name,
            time_budget_seconds=per_backend_budget,
            elapsed_seconds=elapsed,
            autotune_spec={"rows": spec.rows, "cols": spec.cols},
            measurement_spec={"rows": measurement_spec.rows, "cols": measurement_spec.cols},
            dataset_paths={
                "autotune_matrix": str(autotune_dataset.matrix_path),
                "autotune_vector": str(autotune_dataset.vector_path),
                "measurement_matrix": str(measurement_dataset.matrix_path),
                "measurement_vector": str(measurement_dataset.vector_path),
            },
            probe_message=(hardware_inventory.get(backend.name) or {}).get("probe_message"),
        )
        result = backend.run(
            spec,
            autotune_dataset,
            measurement_spec=measurement_spec,
            measurement_dataset=measurement_dataset,
            time_budget_seconds=per_backend_budget,
            force_rebuild=force_rebuild,
        )
        emit_status(
            "method.fmvm.backend.complete",
            status="running",
            method="fixed_matrix_vector_multiplication",
            backend=backend.name,
            available=bool(getattr(result, "available", False)),
            selected_config=getattr(result, "selected_config", None),
            notes=list(getattr(result, "notes", [])),
        )
        backend_results.append(
            result
        )
        runnable_index += 1

    total_elapsed = time.perf_counter() - total_started
    emit_status(
        "method.fmvm.complete",
        status="running",
        method="fixed_matrix_vector_multiplication",
        elapsed_seconds=total_elapsed,
        detected_backends=detected_backends,
    )
    return build_report(
        method="fixed_matrix_vector_multiplication",
        total_elapsed=total_elapsed,
        force_rebuild=force_rebuild,
        autotune_dataset=autotune_dataset,
        measurement_dataset=measurement_dataset,
        autotune_spec=spec,
        measurement_spec=measurement_spec,
        full_runtime_measurement=use_runtime_measurement,
        dataset_was_generated=dataset_was_generated,
        hardware_inventory=hardware_inventory,
        detected_backends=detected_backends,
        backend_results=backend_results,
        backends=backends,
        to_relative_string=to_relative_string,
        root_dir=ROOT_DIR,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper around `run_benchmark()`."""

    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2)
    args.output.write_text(report_text, encoding="utf-8")
    print(report_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
