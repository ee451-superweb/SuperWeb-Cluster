"""Refresh worker GFLOPS estimates while the compute node is idle.

Use this module when a compute node has no in-flight tasks but wants to report
fresh effective performance back to the main node without rerunning full
autotune. It reuses the stored best config and runs a dedicated refresh-sized
benchmark pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from app.constants import METHOD_GEMV, METHOD_CONV2D
from app.compute_resource_policy import resolve_metal_headroom_policy
from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, MethodPerformanceSummary
from compute_node.performance_metrics.gemv.backends import (
    build_backends as build_gemv_backends,
)
from compute_node.performance_metrics.gemv.config import (
    DATASET_DIR as GEMV_DATASET_DIR,
)
from compute_node.performance_metrics.gemv.workloads import (
    build_benchmark_spec as build_gemv_benchmark_spec,
)
from compute_node.performance_metrics.conv2d.backends import (
    build_backends as build_conv2d_backends,
)
from compute_node.performance_metrics.conv2d.config import (
    DATASET_DIR as CONV2D_DATASET_DIR,
)
from compute_node.performance_metrics.conv2d.workloads import get_refresh_spec as get_conv2d_refresh_spec
from compute_node.performance_summary import DEFAULT_RESULT_PATH
from compute_node.input_matrix.gemv import (
    build_dataset_layout as build_gemv_dataset_layout,
    dataset_is_generated as gemv_dataset_is_generated,
)
from compute_node.input_matrix.conv2d import (
    build_dataset_layout as build_conv2d_dataset_layout,
    dataset_is_generated as conv2d_dataset_is_generated,
)

ProgressCallback = Callable[[int, int, str], None]


def _load_result_payload(result_path: Path | None = None) -> dict[str, object]:
    """Use this to read the persisted benchmark summary before idle refresh starts.

    Args: result_path optional override for the benchmark result JSON file.
    Returns: Parsed benchmark payload as a dictionary.
    """
    resolved = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_method_payload(payload: dict[str, object], method: str) -> dict[str, object]:
    """Use this to extract one method block from the persisted benchmark payload.

    Args: payload full benchmark JSON payload and method requested method name.
    Returns: The method-specific dictionary used to refresh one method's performance.
    """
    methods = payload.get("methods")
    if isinstance(methods, dict):
        method_payload = methods.get(method)
        if isinstance(method_payload, dict):
            return method_payload
    raise ValueError(f"benchmark result is missing method payload for {method}")


def _load_best_backend_config(method_payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    """Use this to find the stored best refreshable backend and its best config.

    Args: method_payload one method block from the benchmark result JSON.
    Returns: A tuple of backend name and best-config dictionary for refresh.
    """
    backend_results = method_payload.get("backends")
    if not isinstance(backend_results, dict):
        backend_results = method_payload.get("backend_results")
    if not isinstance(backend_results, dict):
        raise ValueError("benchmark result is missing backends")

    preferred_backend_names: list[str] = []
    best_backend = str(method_payload.get("best_backend") or "")
    if best_backend:
        preferred_backend_names.append(best_backend)
    ranking = method_payload.get("ranking")
    if isinstance(ranking, list):
        preferred_backend_names.extend(str(name) for name in ranking)
    preferred_backend_names.extend(str(name) for name in backend_results)

    for backend_name in preferred_backend_names:
        if backend_name not in {"cpu", "cuda", "metal"}:
            continue
        backend_payload = backend_results.get(backend_name)
        if not isinstance(backend_payload, dict):
            continue
        best_config = backend_payload.get("best_config")
        if isinstance(best_config, dict):
            return backend_name, best_config
    raise ValueError("benchmark result does not contain a refreshable best_config")


def _ensure_gemv_refresh_dataset(backend_name: str, best_config: dict[str, object]) -> None:
    """Use this to validate the GEMV refresh dataset before idle refresh starts.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: ``None`` once the refresh dataset exists locally.
    """
    del best_config
    dataset = build_gemv_dataset_layout(GEMV_DATASET_DIR, prefix="refresh_")
    if not gemv_dataset_is_generated(dataset, build_gemv_benchmark_spec(default_variant="refresh")):
        raise FileNotFoundError(f"GEMV refresh dataset is missing at {dataset.root_dir}")


def _ensure_gemv_refresh_runner(backend_name: str, best_config: dict[str, object]) -> None:
    """Use this to validate the GEMV runner binary before idle refresh starts.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: ``None`` once the chosen backend runner is available locally.
    """
    del best_config
    backend = build_gemv_backends([backend_name])[0]
    if backend_name in {"cpu", "cuda"}:
        backend._resolve_executable_path(force_rebuild=False)
        return
    if backend_name == "metal":
        backend._compile_if_needed(force_rebuild=False)
        return
    raise ValueError(f"unsupported GEMV refresh backend: {backend_name}")


def _ensure_conv2d_refresh_dataset(backend_name: str, best_config: dict[str, object]) -> None:
    """Use this to validate the conv2d refresh dataset before idle refresh starts.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: ``None`` once the refresh dataset exists locally.
    """
    del best_config
    dataset = build_conv2d_dataset_layout(CONV2D_DATASET_DIR, prefix="refresh_")
    if not conv2d_dataset_is_generated(dataset, get_conv2d_refresh_spec(), skip_weight=False):
        raise FileNotFoundError(f"conv2d refresh dataset is missing at {dataset.root_dir}")


def _ensure_conv2d_refresh_runner(backend_name: str, best_config: dict[str, object]) -> None:
    """Use this to validate the spatial runner binary before idle refresh starts.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: ``None`` once the chosen backend runner is available locally.
    """
    del best_config
    backend = build_conv2d_backends([backend_name])[0]
    if backend_name in {"cpu", "cuda"}:
        backend._resolve_executable_path(force_rebuild=False)
        return
    if backend_name == "metal":
        backend._compile_if_needed(force_rebuild=False)
        return
    raise ValueError(f"unsupported spatial refresh backend: {backend_name}")


def _compat_int(best_config: dict[str, object], key: str, default: int) -> int:
    """Use this for runner flags that older and newer backends may omit."""

    value = best_config.get(key)
    if value in (None, ""):
        return default
    return int(value)


def _refresh_gemv_gflops(backend_name: str, best_config: dict[str, object]) -> float:
    """Use this while refreshing idle GEMV performance on the refresh dataset.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: Measured effective GFLOPS for the GEMV method on this worker.
    """
    backend = build_gemv_backends([backend_name])[0]
    spec = build_gemv_benchmark_spec(
        default_variant="refresh",
        accumulation_precision=str(best_config.get("accumulation_precision") or "fp32"),
    )
    dataset = build_gemv_dataset_layout(GEMV_DATASET_DIR, prefix="refresh_")
    if not gemv_dataset_is_generated(dataset, build_gemv_benchmark_spec(default_variant="refresh")):
        raise FileNotFoundError(f"GEMV refresh dataset is missing at {dataset.root_dir}")
    timeout_seconds = max(30.0, spec.zero_score_seconds)

    if backend_name == "cpu":
        executable_path, _ = backend._resolve_executable_path(force_rebuild=False)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            workers=[int(best_config.get("workers") or best_config.get("requested_workers") or 1)],
            tile_sizes=[int(best_config["tile_size"])],
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    elif backend_name == "cuda":
        executable_path, _ = backend._resolve_executable_path(force_rebuild=False)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            transpose_modes=[1 if bool(best_config.get("transpose")) else 0],
            block_sizes=[int(best_config["block_size"])],
            tile_sizes=[int(best_config["tile_size"])],
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    elif backend_name == "metal":
        executable_path, _ = backend._compile_if_needed(force_rebuild=False)
        headroom_policy = resolve_metal_headroom_policy(spec.rows)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            block_sizes=[_compat_int(best_config, "block_size", 256)],
            tile_sizes=[_compat_int(best_config, "tile_size", 1)],
            headroom_fraction=float(best_config.get("headroom_fraction") or headroom_policy.headroom_fraction),
            row_chunk_size=headroom_policy.work_chunk_size,
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported GEMV refresh backend: {backend_name}")

    return float(metrics["measurement_effective_gflops"])


def _refresh_conv2d_gflops(backend_name: str, best_config: dict[str, object]) -> float:
    """Use this while refreshing idle convolution performance on the refresh dataset.

    Args: backend_name selected refresh backend and best_config stored autotune config.
    Returns: Measured effective GFLOPS for the conv2d method.
    """
    backend = build_conv2d_backends([backend_name])[0]
    spec = get_conv2d_refresh_spec()
    dataset = build_conv2d_dataset_layout(CONV2D_DATASET_DIR, prefix="refresh_")
    if not conv2d_dataset_is_generated(dataset, spec, skip_weight=False):
        raise FileNotFoundError(f"conv2d refresh dataset is missing at {dataset.root_dir}")
    timeout_seconds = max(30.0, spec.zero_score_seconds)

    if backend_name == "cpu":
        executable_path, _ = backend._resolve_executable_path(force_rebuild=False)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            workers=[int(best_config.get("workers") or best_config.get("requested_workers") or 1)],
            tile_sizes=[int(best_config["tile_size"])],
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    elif backend_name == "cuda":
        executable_path, _ = backend._resolve_executable_path(force_rebuild=False)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            block_sizes=[int(best_config["block_size"])],
            tile_sizes=[int(best_config["tile_size"])],
            transpose_modes=[1 if bool(best_config.get("transpose")) else 0],
            output_channel_batches=[max(1, min(int(best_config.get("output_channel_batch") or spec.c_out), spec.c_out))],
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    elif backend_name == "metal":
        executable_path, _ = backend._compile_if_needed(force_rebuild=False)
        headroom_policy = resolve_metal_headroom_policy(spec.c_out)
        metrics = backend._run_runner(
            executable_path,
            spec,
            dataset,
            block_sizes=[_compat_int(best_config, "block_size", 256)],
            tile_sizes=[_compat_int(best_config, "tile_size", 16)],
            headroom_fraction=float(best_config.get("headroom_fraction") or headroom_policy.headroom_fraction),
            output_channel_batch=headroom_policy.work_chunk_size,
            autotune_repeats=1,
            measurement_repeats=1,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"unsupported spatial refresh backend: {backend_name}")

    return float(metrics["measurement_effective_gflops"])


def refresh_idle_performance_summary(result_path: Path | None = None) -> ComputePerformanceSummary:
    """Use this when an idle worker needs a fresh abstract performance summary.

    Args: result_path optional benchmark-result path whose best configs should be reused.
    Returns: A ComputePerformanceSummary with refreshed per-method effective GFLOPS.
    """

    payload = _load_result_payload(result_path)
    method_summaries: list[MethodPerformanceSummary] = []

    for method in (METHOD_GEMV, METHOD_CONV2D):
        method_payload = _load_method_payload(payload, method)
        backend_name, best_config = _load_best_backend_config(method_payload)
        effective_gflops = (
            _refresh_gemv_gflops(backend_name, best_config)
            if method == METHOD_GEMV
            else _refresh_conv2d_gflops(backend_name, best_config)
        )
        method_summaries.append(
            MethodPerformanceSummary(
                method=method,
                hardware_count=1,
                ranked_hardware=[
                    ComputeHardwarePerformance(
                        hardware_type=backend_name,
                        effective_gflops=effective_gflops,
                        rank=1,
                    )
                ],
            )
        )

    legacy_view = next(
        summary
        for summary in method_summaries
        if summary.method == METHOD_GEMV
    )
    return ComputePerformanceSummary(
        hardware_count=legacy_view.hardware_count,
        ranked_hardware=list(legacy_view.ranked_hardware),
        method_summaries=method_summaries,
    )


def _emit_validation_progress(
    progress_callback: ProgressCallback | None,
    *,
    step: int,
    total_steps: int,
    description: str,
) -> None:
    """Use this helper to publish human-readable progress during startup checks.

    Args:
        progress_callback: Optional callback supplied by bootstrap logging.
        step: One-based current validation step number.
        total_steps: Total number of validation steps in this pass.
        description: Human-readable description of the current check.

    Returns:
        ``None`` after forwarding the progress event when a callback exists.
    """
    if progress_callback is not None:
        progress_callback(step, total_steps, description)


def validate_idle_refresh_requirements(
    result_path: Path | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Use this before runtime startup to ensure idle refresh can actually run.

    Args:
        result_path: Optional benchmark-result path whose best configs should be reused.
        progress_callback: Optional callback for reporting step-by-step validation progress.

    Returns: ``None`` when the persisted result, refresh datasets, and runners are ready.
    """
    total_steps = 7
    _emit_validation_progress(
        progress_callback,
        step=1,
        total_steps=total_steps,
        description="Loading persisted benchmark result metadata.",
    )
    payload = _load_result_payload(result_path)

    gemv_payload = _load_method_payload(payload, METHOD_GEMV)
    _emit_validation_progress(
        progress_callback,
        step=2,
        total_steps=total_steps,
        description="Selecting refresh backend for gemv.",
    )
    gemv_backend_name, gemv_best_config = _load_best_backend_config(gemv_payload)
    _emit_validation_progress(
        progress_callback,
        step=3,
        total_steps=total_steps,
        description=(
            "Checking gemv refresh input matrix "
            f"for backend {gemv_backend_name}."
        ),
    )
    _ensure_gemv_refresh_dataset(gemv_backend_name, gemv_best_config)
    _emit_validation_progress(
        progress_callback,
        step=4,
        total_steps=total_steps,
        description=(
            "Checking gemv runner binary "
            f"for backend {gemv_backend_name}."
        ),
    )
    _ensure_gemv_refresh_runner(gemv_backend_name, gemv_best_config)

    conv2d_payload = _load_method_payload(payload, METHOD_CONV2D)
    _emit_validation_progress(
        progress_callback,
        step=5,
        total_steps=total_steps,
        description="Selecting refresh backend for conv2d.",
    )
    conv2d_backend_name, conv2d_best_config = _load_best_backend_config(conv2d_payload)
    _emit_validation_progress(
        progress_callback,
        step=6,
        total_steps=total_steps,
        description=(
            "Checking conv2d refresh input matrix "
            f"for backend {conv2d_backend_name}."
        ),
    )
    _ensure_conv2d_refresh_dataset(conv2d_backend_name, conv2d_best_config)
    _emit_validation_progress(
        progress_callback,
        step=7,
        total_steps=total_steps,
        description=(
            "Checking conv2d runner binary "
            f"for backend {conv2d_backend_name}."
        ),
    )
    _ensure_conv2d_refresh_runner(conv2d_backend_name, conv2d_best_config)
