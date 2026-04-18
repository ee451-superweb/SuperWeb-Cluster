"""Assemble raw GEMV benchmark results into the method report schema.

Use this module after GEMV backends finish running so the benchmark can rank
backends, serialize their results, and emit one method-local report payload.
"""

from __future__ import annotations

import platform
import sys
import time

from compute_node.performance_metrics.gemv.models import (
    DEFAULT_ACCUMULATION_PRECISION,
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)


def trial_sort_key(result) -> tuple[float, float]:
    """Order backend best-trials from strongest to weakest.

    Use this when ranking GEMV backends so higher score wins, with lower
    latency breaking ties among equal scores.

    Args:
        result: Backend-result object produced by the benchmark runner.

    Returns:
        A tuple suitable for sorting backend results.
    """

    if result.best_trial is None:
        return (float("-inf"), float("inf"))
    return (result.best_trial.score, -result.best_trial.wall_clock_latency_seconds)


def serialize_backend_result(result, rank_lookup: dict[str, int]) -> dict[str, object]:
    """Convert one backend summary into the JSON layout consumed by ``result.json``.

    Args:
        result: Backend-result object produced by the benchmark runner.
        rank_lookup: Mapping from backend name to overall ranking position.

    Returns:
        The serialized backend section for the GEMV report.
    """

    autotune_trial = result.autotune_trial
    best_trial = result.best_trial
    best_config = None
    autotune_result = None
    best_result = None
    trial_notes: list[str] = []
    if autotune_trial is not None:
        autotune_result = {
            "wall_clock_latency_seconds": autotune_trial.wall_clock_latency_seconds,
            "effective_gflops": autotune_trial.effective_gflops,
            "checksum": autotune_trial.checksum,
            "score": autotune_trial.score,
        }
    if best_trial is not None:
        best_config = dict(result.selected_config or best_trial.config)
        best_result = {
            "wall_clock_latency_seconds": best_trial.wall_clock_latency_seconds,
            "effective_gflops": best_trial.effective_gflops,
            "checksum": best_trial.checksum,
            "score": best_trial.score,
        }
        trial_notes = list(best_trial.notes)

    return {
        "available": result.available,
        "rank": rank_lookup.get(result.backend),
        "best_config": best_config,
        "autotune_result": autotune_result,
        "best_result": best_result,
        "notes": list(result.notes),
        "trial_notes": trial_notes,
    }


def probe_backends(backends: list[object]) -> dict[str, dict[str, object]]:
    """Ask each backend whether it looks usable on this machine before running it.

    Args:
        backends: Backend objects that expose a ``probe()`` method.

    Returns:
        A mapping of backend name to probe availability and message.
    """

    inventory: dict[str, dict[str, object]] = {}
    for backend in backends:
        probe_available, probe_message = backend.probe()
        inventory[str(backend.name)] = {
            "probe_available": bool(probe_available),
            "probe_message": str(probe_message),
        }
    return inventory


def build_report(
    *,
    method: str,
    total_elapsed: float,
    force_rebuild: bool,
    autotune_dataset,
    measurement_dataset,
    autotune_spec,
    measurement_spec,
    workload_mode: str,
    autotune_dataset_variant: str,
    measurement_dataset_variant: str,
    full_runtime_measurement: bool,
    dataset_was_generated: bool,
    hardware_inventory: dict[str, dict[str, object]],
    detected_backends: list[str],
    backend_results: list[object],
    backends: list[object],
    to_relative_string,
    root_dir,
) -> dict[str, object]:
    """Assemble the GEMV JSON-serializable benchmark summary.

    Use this once all GEMV backends have finished so the method-local report has
    one stable schema regardless of which backends were available.

    Args:
        method: Logical method name.
        total_elapsed: Total wall-clock time spent benchmarking the method.
        force_rebuild: Whether backend binaries were forced to rebuild.
        autotune_dataset: Dataset layout used during autotune.
        measurement_dataset: Dataset layout used during final measurement.
        autotune_spec: Benchmark spec used during autotune.
        measurement_spec: Benchmark spec used during final measurement.
        workload_mode: Requested workload mode.
        autotune_dataset_variant: Dataset variant used during autotune.
        measurement_dataset_variant: Dataset variant used during final measurement.
        full_runtime_measurement: Whether the run used small autotune plus large measurement.
        dataset_was_generated: Whether missing datasets were generated on demand.
        hardware_inventory: Probe results keyed by backend name.
        detected_backends: Backends that passed probe.
        backend_results: Backend-result objects produced during the run.
        backends: Backend objects considered by the run.
        to_relative_string: Path-normalization helper for report output.
        root_dir: Project-relative base directory for path normalization.

    Returns:
        The GEMV method report dictionary.
    """

    ranked_backend_results = [
        result for result in sorted(backend_results, key=trial_sort_key, reverse=True) if result.best_trial is not None
    ]
    rank_lookup = {result.backend: rank for rank, result in enumerate(ranked_backend_results, start=1)}
    backend_result_map = {
        result.backend: serialize_backend_result(result, rank_lookup) for result in backend_results
    }
    ranking = [result.backend for result in ranked_backend_results]
    best_backend = ranking[0] if ranking else None

    return {
        "schema_version": 2,
        "method": method,
        "generated_at_unix": time.time(),
        "benchmark_elapsed_seconds": total_elapsed,
        "workload": {
            "workload_mode": workload_mode,
            "autotune": {
                "name": autotune_spec.name,
                "rows": autotune_spec.rows,
                "cols": autotune_spec.cols,
            },
            "measurement": {
                "name": measurement_spec.name,
                "rows": measurement_spec.rows,
                "cols": measurement_spec.cols,
            },
            "autotune_repeats": DEFAULT_AUTOTUNE_REPEATS,
            "measurement_repeats": (
                1 if measurement_dataset_variant == "large" else DEFAULT_MEASUREMENT_REPEATS
            ),
            "selection_metric": "autotune_average_latency",
            "reported_metric": "measurement_average_latency",
            "force_rebuild": force_rebuild,
            "input_dtype": "fp32",
            "output_dtype": "fp32",
            "accumulation_precision": autotune_spec.accumulation_precision or DEFAULT_ACCUMULATION_PRECISION,
            "cross_backend_validation": "fp32_tolerance",
            "autotune_dataset_variant": autotune_dataset_variant,
            "measurement_dataset_variant": measurement_dataset_variant,
            "full_runtime_measurement": full_runtime_measurement,
        },
        "host": {
            "platform": sys.platform,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "dataset": {
            "root_dir": to_relative_string(autotune_dataset.root_dir, start=root_dir),
            "artifacts": {
                "autotune_matrix": to_relative_string(autotune_dataset.matrix_path, start=root_dir),
                "autotune_vector": to_relative_string(autotune_dataset.vector_path, start=root_dir),
                "measurement_matrix": to_relative_string(measurement_dataset.matrix_path, start=root_dir),
                "measurement_vector": to_relative_string(measurement_dataset.vector_path, start=root_dir),
            },
            "shape": {
                "autotune": {
                    "rows": autotune_spec.rows,
                    "cols": autotune_spec.cols,
                },
                "measurement": {
                    "rows": measurement_spec.rows,
                    "cols": measurement_spec.cols,
                },
            },
            "bytes": {
                "autotune_matrix": autotune_spec.matrix_bytes,
                "autotune_vector": autotune_spec.vector_bytes,
                "measurement_matrix": measurement_spec.matrix_bytes,
                "measurement_vector": measurement_spec.vector_bytes,
            },
            "dataset_was_generated": dataset_was_generated,
        },
        "hardware_inventory": hardware_inventory,
        "detected_backends": detected_backends,
        "usable_backends": [result.backend for result in backend_results if result.available],
        "backends_considered": [backend.name for backend in backends],
        "best_backend": best_backend,
        "ranking": ranking,
        "backend_results": backend_result_map,
    }
