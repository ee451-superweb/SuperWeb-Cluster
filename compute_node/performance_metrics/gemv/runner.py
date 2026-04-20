"""Run the GEMV benchmark and emit the raw method-local report.

Use this module when the GEMV method should benchmark its backends directly,
either from the top-level runner or through the GEMV-only wrapper CLI.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode

enable_utf8_mode()

from compute_node.input_matrix.gemv import build_dataset_layout, dataset_prefix_for_size
from compute_node.performance_metrics.device_overview import detect_cpu_name
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.performance_metrics.gemv.config import (
    DATASET_DIR as METHOD_DATASET_DIR,
    GENERATE_SCRIPT_PATH as METHOD_GENERATE_SCRIPT_PATH,
    RESULT_PATH as METHOD_RESULT_PATH,
)
from compute_node.performance_metrics.gemv.backends import build_backends
from compute_node.performance_metrics.gemv.dataset_runner import (
    generate_dataset_if_missing,
    resolve_dataset_dir,
)
from compute_node.performance_metrics.gemv.reporting import (
    build_report,
    probe_backends,
)
from compute_node.performance_metrics.gemv.workloads import build_benchmark_spec
from compute_node.performance_metrics.path_utils import to_relative_string
from compute_node.performance_metrics.workload_modes import (
    BENCHMARK_WORKLOAD_MODE_CHOICES,
    WORKLOAD_MODE_CUSTOM,
    WORKLOAD_MODE_FULL,
    WORKLOAD_MODE_LARGE,
    WORKLOAD_MODE_MID,
    WORKLOAD_MODE_MEDIUM,
    WORKLOAD_MODE_SMALL,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = METHOD_DATASET_DIR
DEFAULT_OUTPUT_PATH = METHOD_RESULT_PATH
GENERATE_SCRIPT_PATH = METHOD_GENERATE_SCRIPT_PATH
_QUOTED_HARDWARE_PATTERN = re.compile(r"'([^']+)'")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the GEMV benchmark runner.

    Args:
        None.

    Returns:
        The configured ``ArgumentParser`` for the GEMV benchmark entrypoint.
    """

    parser = argparse.ArgumentParser(description="Benchmark GEMV backends.")
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
        "--verbose",
        action="store_true",
        help="Stream additional per-backend configuration and selection details to stdout.",
    )
    parser.add_argument(
        "--accumulation-precision",
        choices=("fp32", "fp64_accumulate"),
        default="fp32",
        help="Numeric accumulation mode. Default: fp32. fp64_accumulate is slower but can reduce tiny cross-backend drift.",
    )
    parser.add_argument("--rows", type=int, help="Optional small-size override for the matrix row count.")
    parser.add_argument("--cols", type=int, help="Optional small-size override for the matrix column count.")
    parser.add_argument(
        "--workload-mode",
        choices=BENCHMARK_WORKLOAD_MODE_CHOICES,
        default=WORKLOAD_MODE_FULL,
        help="Benchmark only the small, mid, or large dataset, or run the default small->large flow.",
    )
    return parser


def _has_custom_workload(args: argparse.Namespace) -> bool:
    """Return whether CLI overrides requested a custom GEMV shape.

    Args:
        args: Parsed benchmark CLI arguments.

    Returns:
        ``True`` when rows or cols were overridden explicitly.
    """
    return args.rows is not None or args.cols is not None


def _resolve_workload_plan(
    args: argparse.Namespace,
    *,
    accumulation_precision: str,
) -> tuple[str, str, str, object, object]:
    """Resolve autotune and measurement plans from CLI workload choices.

    Use this before dataset generation so the runner knows which benchmark spec
    and dataset variant to use for autotune and final measurement.

    Args:
        args: Parsed benchmark CLI arguments.
        accumulation_precision: Numeric accumulation mode for GEMV.

    Returns:
        A tuple containing workload mode, autotune variant, measurement variant,
        autotune spec, and measurement spec.
    """
    requested_mode = getattr(args, "workload_mode", WORKLOAD_MODE_FULL)
    if _has_custom_workload(args):
        custom_spec = build_benchmark_spec(
            default_variant="small",
            rows=args.rows,
            cols=args.cols,
            accumulation_precision=accumulation_precision,
        )
        return (
            WORKLOAD_MODE_CUSTOM,
            WORKLOAD_MODE_CUSTOM,
            WORKLOAD_MODE_CUSTOM,
            custom_spec,
            custom_spec,
        )

    if requested_mode == WORKLOAD_MODE_SMALL:
        small_spec = build_benchmark_spec(
            default_variant="small",
            accumulation_precision=accumulation_precision,
        )
        return (
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            small_spec,
            small_spec,
        )

    if requested_mode == WORKLOAD_MODE_MEDIUM:
        mid_spec = build_benchmark_spec(
            default_variant="mid",
            accumulation_precision=accumulation_precision,
        )
        return (
            WORKLOAD_MODE_MID,
            WORKLOAD_MODE_MID,
            WORKLOAD_MODE_MID,
            mid_spec,
            mid_spec,
        )

    if requested_mode == WORKLOAD_MODE_LARGE:
        large_spec = build_benchmark_spec(
            default_variant="large",
            accumulation_precision=accumulation_precision,
        )
        return (
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            large_spec,
            large_spec,
        )

    small_spec = build_benchmark_spec(
        default_variant="small",
        accumulation_precision=accumulation_precision,
    )
    large_spec = build_benchmark_spec(
        default_variant="large",
        accumulation_precision=accumulation_precision,
    )
    return (
        WORKLOAD_MODE_FULL,
        WORKLOAD_MODE_SMALL,
        WORKLOAD_MODE_LARGE,
        small_spec,
        large_spec,
    )


def _hardware_label_for_backend(backend_name: str, probe_message: str) -> str:
    """Extract one operator-facing hardware label from a probe result.

    Args:
        backend_name: Logical backend name such as ``cpu`` or ``cuda``.
        probe_message: Human-readable backend probe message.

    Returns:
        A short hardware label suitable for benchmark progress output.
    """

    quoted_match = _QUOTED_HARDWARE_PATTERN.search(probe_message)
    if quoted_match:
        return quoted_match.group(1).strip()
    if backend_name == "cpu":
        return detect_cpu_name().strip() or platform.machine() or "local CPU"
    if backend_name == "cuda":
        return "NVIDIA GPU"
    if backend_name == "dx12":
        return "D3D12 adapter"
    if backend_name == "metal":
        return "Metal device"
    return backend_name


def _phase_label(dataset_variant: str, spec) -> str:
    """Return one concise human-facing GEMV phase label."""

    return f"{dataset_variant} ({spec.rows}x{spec.cols})"


def _measurement_phase_title(
    autotune_dataset_variant: str,
    measurement_dataset_variant: str,
) -> str:
    """Return the stdout title for the GEMV post-autotune phase."""

    if measurement_dataset_variant != autotune_dataset_variant:
        return "Full run stage"
    return "Final measurement stage"


def _format_selected_config(config: dict[str, object] | None) -> str:
    """Render one selected GEMV config as a compact stdout/log string."""

    if not config:
        return "<none>"

    parts: list[str] = []
    for key, value in config.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, float):
            rendered = f"{value:g}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the GEMV benchmark and return the JSON-serializable report.

    Use this from the top-level benchmark runner or tests when the caller wants
    the GEMV method report as a Python dictionary.

    Args:
        args: Parsed benchmark CLI arguments.

    Returns:
        The raw GEMV method report dictionary.
    """

    emit_status(
        "method.gemv.start",
        status="running",
        method="gemv",
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        workload_mode=getattr(args, "workload_mode", WORKLOAD_MODE_FULL),
        rows=args.rows,
        cols=args.cols,
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
    )
    backends = build_backends(args.backend)
    force_rebuild = bool(getattr(args, "rebuild", False))
    verbose = bool(getattr(args, "verbose", False))
    hardware_inventory = probe_backends(backends)
    detected_backends = [
        backend.name
        for backend in backends
        if bool((hardware_inventory.get(backend.name) or {}).get("probe_available"))
    ]

    accumulation_precision = getattr(args, "accumulation_precision", "fp32")
    (
        workload_mode,
        autotune_dataset_variant,
        measurement_dataset_variant,
        spec,
        measurement_spec,
    ) = _resolve_workload_plan(
        args,
        accumulation_precision=accumulation_precision,
    )
    dataset_dir = resolve_dataset_dir(args, spec, default_dataset_dir=DEFAULT_DATASET_DIR)
    small_dataset = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("small"))
    mid_dataset = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("mid"))
    large_dataset = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("large"))
    if autotune_dataset_variant == WORKLOAD_MODE_LARGE:
        autotune_dataset = large_dataset
    elif autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        autotune_dataset = mid_dataset
    else:
        autotune_dataset = small_dataset
    if measurement_dataset_variant == WORKLOAD_MODE_LARGE:
        measurement_dataset = large_dataset
    elif measurement_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        measurement_dataset = mid_dataset
    else:
        measurement_dataset = autotune_dataset
    dataset_was_generated = False
    if detected_backends:
        emit_status(
            "method.gemv.dataset.check",
            status="running",
            method="gemv",
            dataset_dir=str(dataset_dir),
            workload_mode=workload_mode,
            generate_small_dataset=autotune_dataset_variant != WORKLOAD_MODE_LARGE,
            generate_mid_dataset=(
                autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
                or measurement_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
            ),
            generate_large_dataset=measurement_dataset_variant == WORKLOAD_MODE_LARGE,
        )
        dataset_was_generated = generate_dataset_if_missing(
            dataset_dir,
            spec.rows,
            spec.cols,
            generate_small_dataset=autotune_dataset_variant != WORKLOAD_MODE_LARGE,
            generate_mid_dataset=(
                autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
                or measurement_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
            ),
            generate_large_dataset=measurement_dataset_variant == WORKLOAD_MODE_LARGE,
            root_dir=ROOT_DIR,
            generate_script_path=GENERATE_SCRIPT_PATH,
        )

    total_started = time.perf_counter()
    backend_results = []
    runnable_backends = [backend for backend in backends if backend.name in detected_backends]
    runnable_index = 0
    if backends:
        print("\n=== Benchmarking GEMV Backends ===", flush=True)
        print(
            f"    Autotune stage: {_phase_label(autotune_dataset_variant, spec)}",
            flush=True,
        )
        if verbose:
            print(
                f"[gemv verbose] workload_mode={workload_mode} "
                f"autotune_variant={autotune_dataset_variant} "
                f"measurement_variant={measurement_dataset_variant} "
                f"force_rebuild={force_rebuild} "
                f"detected_backends={detected_backends} "
                f"accumulation_precision={accumulation_precision}",
                flush=True,
            )
    for backend in backends:
        probe_message = str((hardware_inventory.get(backend.name) or {}).get("probe_message") or "")
        hardware_name = _hardware_label_for_backend(backend.name, probe_message)
        if backend.name not in detected_backends:
            print(
                f" -> Skipping {backend.name} on {hardware_name}: "
                f"{probe_message or 'probe unavailable'}",
                flush=True,
            )
            emit_status(
                "method.gemv.backend.skipped",
                status="running",
                method="gemv",
                backend=backend.name,
                hardware_name=hardware_name,
                reason="probe_unavailable",
                probe_message=probe_message,
            )
            backend_results.append(
                backend.run(
                    spec,
                    autotune_dataset,
                    measurement_spec=measurement_spec,
                    measurement_dataset=measurement_dataset,
                    time_budget_seconds=1.0,
                    force_rebuild=force_rebuild,
                    verbose=verbose,
                )
            )
            continue

        elapsed = time.perf_counter() - total_started
        remaining = max(args.time_budget - elapsed, 1.0)
        per_backend_budget = remaining / max(len(runnable_backends) - runnable_index, 1)
        print(f" -> Benchmarking {backend.name} on {hardware_name}...", flush=True)
        print(
            f"    Autotune stage on {hardware_name}: "
            f"{_phase_label(autotune_dataset_variant, spec)}",
            flush=True,
        )

        def _announce_phase(
            phase: str,
            selected_config: dict[str, object] | None = None,
        ) -> None:
            if phase != "final_measurement":
                return
            config_text = _format_selected_config(selected_config)
            print(
                f"    Autotune best config on {hardware_name}: {config_text}",
                flush=True,
            )
            print(
                f"    {_measurement_phase_title(autotune_dataset_variant, measurement_dataset_variant)} "
                f"on {hardware_name}: {_phase_label(measurement_dataset_variant, measurement_spec)} "
                f"using {config_text}",
                flush=True,
            )
            emit_status(
                "method.gemv.backend.autotune_selected",
                status="running",
                method="gemv",
                backend=backend.name,
                hardware_name=hardware_name,
                selected_config=selected_config,
                selected_config_text=config_text,
                measurement_phase=_measurement_phase_title(
                    autotune_dataset_variant,
                    measurement_dataset_variant,
                ),
            )
        emit_status(
            "method.gemv.backend.start",
            status="running",
            method="gemv",
            backend=backend.name,
            hardware_name=hardware_name,
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
            probe_message=probe_message,
        )
        result = backend.run(
            spec,
            autotune_dataset,
            measurement_spec=measurement_spec,
            measurement_dataset=measurement_dataset,
            time_budget_seconds=per_backend_budget,
            force_rebuild=force_rebuild,
            phase_callback=_announce_phase,
            verbose=verbose,
        )
        emit_status(
            "method.gemv.backend.complete",
            status="running",
            method="gemv",
            backend=backend.name,
            hardware_name=hardware_name,
            available=bool(getattr(result, "available", False)),
            selected_config=getattr(result, "selected_config", None),
            notes=list(getattr(result, "notes", [])),
        )
        if bool(getattr(result, "available", False)) and getattr(result, "best_trial", None) is not None:
            print(
                f"    Available on {hardware_name}. Performance: "
                f"{result.best_trial.effective_gflops:.2f} GFLOPS",
                flush=True,
            )
        else:
            note_list = list(getattr(result, "notes", []))
            print(
                f"    Not available on {hardware_name}: "
                f"{note_list[0] if note_list else 'No details'}",
                flush=True,
            )
        backend_results.append(
            result
        )
        runnable_index += 1

    total_elapsed = time.perf_counter() - total_started
    emit_status(
        "method.gemv.complete",
        status="running",
        method="gemv",
        elapsed_seconds=total_elapsed,
        detected_backends=detected_backends,
    )
    return build_report(
        method="gemv",
        total_elapsed=total_elapsed,
        force_rebuild=force_rebuild,
        autotune_dataset=autotune_dataset,
        measurement_dataset=measurement_dataset,
        autotune_spec=spec,
        measurement_spec=measurement_spec,
        workload_mode=workload_mode,
        autotune_dataset_variant=autotune_dataset_variant,
        measurement_dataset_variant=measurement_dataset_variant,
        full_runtime_measurement=workload_mode == WORKLOAD_MODE_FULL,
        dataset_was_generated=dataset_was_generated,
        hardware_inventory=hardware_inventory,
        detected_backends=detected_backends,
        backend_results=backend_results,
        backends=backends,
        to_relative_string=to_relative_string,
        root_dir=ROOT_DIR,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper around ``run_benchmark()``.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code ``0`` on success.
    """

    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2)
    args.output.write_text(report_text, encoding="utf-8")
    best_backend = str(report.get("best_backend") or "")
    backend_results = report.get("backend_results", {})
    best_result = None
    if best_backend and isinstance(backend_results, dict):
        best_backend_report = backend_results.get(best_backend)
        if isinstance(best_backend_report, dict):
            best_result = best_backend_report.get("best_result")
    effective_gflops = None
    if isinstance(best_result, dict):
        effective_gflops = best_result.get("effective_gflops")
    print("\nBenchmark Summary:", flush=True)
    if best_backend and effective_gflops is not None:
        print(
            f"[benchmark] gemv best result: "
            f"backend={best_backend} effective_gflops={float(effective_gflops):.2f}",
            flush=True,
        )
    else:
        print("[benchmark] gemv best result: no available backend", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
