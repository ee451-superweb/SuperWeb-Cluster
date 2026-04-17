"""Run the spatial-convolution benchmark and emit the raw method report.

Use this module when the spatial benchmark should benchmark its backends
directly, either from the top-level runner or as a standalone CLI.
"""

from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.constants import (
    DEFAULT_SPATIAL_CUDA_COOLDOWN_MS,
    DEFAULT_SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE,
)
from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.performance_metrics.spatial_convolution.backends import (
    build_backends,
)
from compute_node.performance_metrics.spatial_convolution.backends import cuda_backend
from compute_node.performance_metrics.spatial_convolution.config import (
    DATASET_DIR as METHOD_DATASET_DIR,
    GENERATE_SCRIPT_PATH as METHOD_GENERATE_SCRIPT_PATH,
    RESULT_PATH as METHOD_RESULT_PATH,
)
from compute_node.performance_metrics.spatial_convolution.dataset import (
    build_dataset_layout,
    dataset_is_generated,
)
from compute_node.performance_metrics.spatial_convolution.models import (
    BenchmarkSpec,
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)
from compute_node.performance_metrics.spatial_convolution.workloads import (
    build_benchmark_spec,
    get_medium_spec,
    get_runtime_spec,
    get_test_spec,
)
from compute_node.performance_metrics.workload_modes import (
    BENCHMARK_WORKLOAD_MODE_CHOICES,
    WORKLOAD_MODE_CUSTOM,
    WORKLOAD_MODE_FULL,
    WORKLOAD_MODE_LARGE,
    WORKLOAD_MODE_MEDIUM,
    WORKLOAD_MODE_SMALL,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the spatial benchmark entrypoint.

    Args:
        None.

    Returns:
        The configured ``ArgumentParser`` for the spatial benchmark CLI.
    """
    parser = argparse.ArgumentParser(description="Benchmark backends.")
    parser.add_argument("--backend", action="append", default=None)
    parser.add_argument("--dataset-dir", type=Path, default=METHOD_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=METHOD_RESULT_PATH)
    parser.add_argument("--role", choices=("compute", "main"), default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--cin", type=int)
    parser.add_argument("--cout", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pad", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--workload-mode",
        choices=BENCHMARK_WORKLOAD_MODE_CHOICES,
        default=WORKLOAD_MODE_FULL,
        help="Benchmark only the small, medium, or large dataset, or run the default small->large flow.",
    )
    parser.add_argument("--output-channel-batch", type=int)
    parser.add_argument("--output-channel-batch-scale", type=float)
    parser.add_argument("--cooldown-ms", type=float)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def _apply_cuda_cli_overrides(args: argparse.Namespace) -> None:
    """Apply optional CLI overrides for the spatial CUDA benchmark backend.

    Use this before running the benchmark so the CUDA backend sees the CLI's
    temporary autotune and cooldown overrides for this process only.

    Args:
        args: Parsed spatial benchmark CLI arguments.

    Returns:
        ``None`` after module-level CUDA overrides have been updated.
    """

    cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE = None
    cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE = DEFAULT_SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE
    cuda_backend.SPATIAL_CUDA_COOLDOWN_MS = DEFAULT_SPATIAL_CUDA_COOLDOWN_MS

    if args.output_channel_batch is not None:
        if args.output_channel_batch <= 0:
            raise ValueError("--output-channel-batch must be a positive integer")
        cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE = (args.output_channel_batch,)
    if args.output_channel_batch_scale is not None:
        if args.output_channel_batch_scale <= 0.0:
            raise ValueError("--output-channel-batch-scale must be positive")
        cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE = args.output_channel_batch_scale
    if args.cooldown_ms is not None:
        if args.cooldown_ms < 0:
            raise ValueError("--cooldown-ms must be non-negative")
        cuda_backend.SPATIAL_CUDA_COOLDOWN_MS = args.cooldown_ms

def _generate_if_needed(
    dataset_dir: Path,
    spec,
    role: str,
    *,
    generate_small_dataset: bool,
    generate_medium_dataset: bool,
    generate_large_dataset: bool,
) -> None:
    """Generate datasets only when the current benchmark flow actually needs them.

    Use this before backend runs so the requested small, medium, and large
    datasets exist without regenerating variants that are not needed.

    Args:
        dataset_dir: Directory that should hold the generated datasets.
        spec: Resolved autotune benchmark specification.
        role: Dataset-generation role such as ``compute`` or ``main``.
        generate_small_dataset: Whether the small dataset is required.
        generate_medium_dataset: Whether the medium dataset is required.
        generate_large_dataset: Whether the large dataset is required.

    Returns:
        ``None`` after the required dataset files exist.
    """

    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    medium_layout = build_dataset_layout(dataset_dir, prefix="medium_")
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")
    runtime_spec = get_runtime_spec()
    medium_spec = get_medium_spec()

    has_test = dataset_is_generated(test_layout, spec, skip_weight=False) if generate_small_dataset else True
    has_medium = dataset_is_generated(medium_layout, medium_spec, skip_weight=False) if generate_medium_dataset else True
    has_runtime = (
        dataset_is_generated(runtime_layout, runtime_spec, skip_weight=False)
        if generate_large_dataset
        else True
    )

    if has_test and has_medium and has_runtime:
        return

    script = METHOD_GENERATE_SCRIPT_PATH
    cmd = [
        str(active_python_path()),
        str(script),
        "--output-dir",
        str(dataset_dir),
        "--role",
        role,
        "--h",
        str(spec.h),
        "--w",
        str(spec.w),
        "--cin",
        str(spec.c_in),
        "--cout",
        str(spec.c_out),
        "--k",
        str(spec.k),
        "--pad",
        str(spec.pad),
        "--stride",
        str(spec.stride),
    ]
    if not generate_small_dataset:
        cmd.append("--skip-small")
    if not generate_medium_dataset:
        cmd.append("--skip-medium")
    if generate_large_dataset:
        cmd.append("--include-runtime-weight")
    else:
        cmd.append("--skip-large")
    emit_status(
        "method.spatial_convolution.dataset.generate.start",
        status="running",
        method="spatial_convolution",
        dataset_dir=str(dataset_dir),
        role=role,
        generate_small_dataset=generate_small_dataset,
        generate_medium_dataset=generate_medium_dataset,
        generate_large_dataset=generate_large_dataset,
        command=cmd,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(cmd, check=True)
    emit_status(
        "method.spatial_convolution.dataset.generate.complete",
        status="running",
        method="spatial_convolution",
        dataset_dir=str(dataset_dir),
        role=role,
        generate_small_dataset=generate_small_dataset,
        generate_medium_dataset=generate_medium_dataset,
        generate_large_dataset=generate_large_dataset,
    )


def _serialize_backend_result(result: dict, rank: int | None) -> dict:
    """Convert one raw backend result into the report-friendly layout.

    Args:
        result: Raw backend result dictionary.
        rank: Ranking position for the backend, if available.

    Returns:
        The serialized backend result stored in ``result.json``.
    """
    autotune_trial = result.get("autotune_trial")
    best_trial = result.get("best_trial")
    selected_config = result.get("selected_config")

    serialized = dict(result)
    serialized["rank"] = rank
    serialized["best_config"] = selected_config
    serialized["autotune_result"] = autotune_trial
    serialized["best_result"] = best_trial
    return serialized


def _has_custom_workload(args: argparse.Namespace) -> bool:
    """Return whether CLI overrides requested a custom Conv2D shape.

    Args:
        args: Parsed benchmark CLI arguments.

    Returns:
        ``True`` when any shape field was overridden explicitly.
    """
    return any(
        getattr(args, field) is not None
        for field in ("h", "w", "cin", "cout", "k", "pad", "stride")
    )


def _spec_to_dict(spec) -> dict:
    """Serialize one benchmark spec into a JSON-friendly dictionary.

    Args:
        spec: Spatial benchmark specification object.

    Returns:
        A plain dictionary describing the workload dimensions.
    """
    return {
        "name": spec.name,
        "h": spec.h,
        "w": spec.w,
        "c_in": spec.c_in,
        "c_out": spec.c_out,
        "k": spec.k,
        "pad": spec.pad,
        "stride": spec.stride,
        "ideal_seconds": spec.ideal_seconds,
        "zero_score_seconds": spec.zero_score_seconds,
    }


def _final_measurement_repeats(use_runtime_measurement: bool) -> int:
    """Choose the repeat count for the final measurement phase.

    Args:
        use_runtime_measurement: Whether the final phase uses the large runtime workload.

    Returns:
        The repeat count to use for the final measurement phase.
    """
    return 1 if use_runtime_measurement else DEFAULT_MEASUREMENT_REPEATS


def _resolve_workload_plan(args: argparse.Namespace) -> tuple[str, str, str, BenchmarkSpec, BenchmarkSpec]:
    """Resolve autotune and measurement plans from CLI workload choices.

    Use this before dataset generation and backend runs so the benchmark knows
    which spec and dataset variant belong to each phase.

    Args:
        args: Parsed spatial benchmark CLI arguments.

    Returns:
        A tuple containing workload mode, autotune variant, measurement variant,
        autotune spec, and measurement spec.
    """
    if _has_custom_workload(args):
        custom_spec = build_benchmark_spec(
            h=args.h,
            w=args.w,
            c_in=args.cin,
            c_out=args.cout,
            k=args.k,
            pad=args.pad,
            stride=args.stride,
        )
        return (
            WORKLOAD_MODE_CUSTOM,
            WORKLOAD_MODE_CUSTOM,
            WORKLOAD_MODE_CUSTOM,
            custom_spec,
            custom_spec,
        )

    requested_mode = getattr(args, "workload_mode", WORKLOAD_MODE_FULL)
    if requested_mode == WORKLOAD_MODE_SMALL:
        small_spec = get_test_spec()
        return (
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            small_spec,
            small_spec,
        )
    if requested_mode == WORKLOAD_MODE_MEDIUM:
        medium_spec = get_medium_spec()
        return (
            WORKLOAD_MODE_MEDIUM,
            WORKLOAD_MODE_MEDIUM,
            WORKLOAD_MODE_MEDIUM,
            medium_spec,
            medium_spec,
        )
    if requested_mode == WORKLOAD_MODE_LARGE:
        large_spec = get_runtime_spec()
        return (
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            large_spec,
            large_spec,
        )

    small_spec = get_test_spec()
    large_spec = get_runtime_spec()
    return (
        WORKLOAD_MODE_FULL,
        WORKLOAD_MODE_SMALL,
        WORKLOAD_MODE_LARGE,
        small_spec,
        large_spec,
    )


def _backend_diagnostic_context(backend, spec: BenchmarkSpec) -> dict[str, object] | None:
    """Ask a backend for optional diagnostic context for status logging.

    Args:
        backend: Backend object that may expose ``diagnostic_context``.
        spec: Benchmark specification currently being executed.

    Returns:
        A diagnostic-context dictionary, or ``None`` when unavailable.
    """
    builder = getattr(backend, "diagnostic_context", None)
    if not callable(builder):
        return None
    try:
        payload = builder(spec)
    except TypeError:
        payload = builder()
    return payload if isinstance(payload, dict) else None

def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the spatial benchmark and return the raw method report.

    Use this from the top-level benchmark runner or tests when the caller wants
    the spatial benchmark result as a Python dictionary.

    Args:
        args: Parsed spatial benchmark CLI arguments.

    Returns:
        The raw spatial-convolution method report dictionary.
    """
    _apply_cuda_cli_overrides(args)
    workload_mode, autotune_dataset_variant, measurement_dataset_variant, spec, measurement_spec = _resolve_workload_plan(args)
    emit_status(
        "method.spatial_convolution.start",
        status="running",
        method="spatial_convolution",
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        workload_mode=getattr(args, "workload_mode", WORKLOAD_MODE_FULL),
        effective_workload_mode=workload_mode,
        role=args.role,
        h=args.h,
        w=args.w,
        cin=args.cin,
        cout=args.cout,
        k=args.k,
        pad=args.pad,
        stride=args.stride,
        output_channel_batch_candidates=cuda_backend._candidate_output_channel_batches(spec),
        output_channel_batch_scale=cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE,
        cooldown_ms=cuda_backend.SPATIAL_CUDA_COOLDOWN_MS,
    )
    dataset_dir = Path(args.dataset_dir).resolve()
    generate_small_dataset = autotune_dataset_variant != WORKLOAD_MODE_LARGE
    generate_medium_dataset = (
        autotune_dataset_variant == WORKLOAD_MODE_MEDIUM
        or measurement_dataset_variant == WORKLOAD_MODE_MEDIUM
    )
    generate_large_dataset = measurement_dataset_variant == WORKLOAD_MODE_LARGE

    # 1. 确保 benchmark 实际需要的数据存在
    _generate_if_needed(
        dataset_dir,
        spec,
        args.role,
        generate_small_dataset=generate_small_dataset,
        generate_medium_dataset=generate_medium_dataset,
        generate_large_dataset=generate_large_dataset,
    )

    # 2. 探测并运行硬件测试
    backends = build_backends(args.backend)
    backend_results = {}
    hardware_inventory = {}

    print(f"\n=== Benchmarking Backends for {args.role.upper()} NODE ===")
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    medium_layout = build_dataset_layout(dataset_dir, prefix="medium_")
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")
    if autotune_dataset_variant == WORKLOAD_MODE_LARGE:
        autotune_layout = runtime_layout
    elif autotune_dataset_variant == WORKLOAD_MODE_MEDIUM:
        autotune_layout = medium_layout
    else:
        autotune_layout = test_layout
    if measurement_dataset_variant == WORKLOAD_MODE_LARGE:
        measurement_layout = runtime_layout
    elif measurement_dataset_variant == WORKLOAD_MODE_MEDIUM:
        measurement_layout = medium_layout
    else:
        measurement_layout = autotune_layout

    for b in backends:
        print(f" -> Probing {b.name}...")
        probe_available, probe_message = b.probe()
        emit_status(
            "method.spatial_convolution.backend.probe",
            status="running",
            method="spatial_convolution",
            backend=b.name,
            probe_available=bool(probe_available),
            probe_message=str(probe_message),
            diagnostic_context=_backend_diagnostic_context(b, spec),
        )
        hardware_inventory[b.name] = {
            "probe_available": bool(probe_available),
            "probe_message": str(probe_message),
        }
        emit_status(
            "method.spatial_convolution.backend.start",
            status="running",
            method="spatial_convolution",
            backend=b.name,
            autotune_spec=_spec_to_dict(spec),
            measurement_spec=_spec_to_dict(measurement_spec),
            dataset_paths={
                "autotune_input": str(autotune_layout.input_path),
                "autotune_weight": str(autotune_layout.weight_path),
                "measurement_input": str(measurement_layout.input_path),
                "measurement_weight": str(measurement_layout.weight_path),
            },
            probe_available=bool(probe_available),
            probe_message=str(probe_message),
            diagnostic_context=_backend_diagnostic_context(b, spec),
            force_rebuild=bool(args.rebuild),
        )
        result = b.run(
            spec,
            autotune_layout,
            measurement_spec=measurement_spec,
            measurement_dataset=measurement_layout,
            time_budget_seconds=measurement_spec.zero_score_seconds,
            force_rebuild=args.rebuild,
        )
        res = result.to_dict() if hasattr(result, 'to_dict') else result
        backend_results[b.name] = res
        emit_status(
            "method.spatial_convolution.backend.complete",
            status="running",
            method="spatial_convolution",
            backend=b.name,
            available=bool(res.get("available")),
            selected_config=res.get("selected_config") or res.get("best_config"),
            notes=res.get("notes", []),
        )

        if res.get("available"):
            best = res.get("best_trial") or res.get("best_result") or {}
            gflops = best.get("effective_gflops", 0)
            print(f"    Available. Performance: {gflops:.2f} GFLOPS")
        else:
            notes = res.get("notes", ["No details"])
            print(f"    Not available: {notes[0] if notes else 'No details'}")

    # 3. 生成排名
    ranking = sorted(
        [k for k, v in backend_results.items() if v.get("available")],
        key=lambda x: (backend_results[x].get("best_trial") or backend_results[x].get("best_result") or {}).get("effective_gflops", 0),
        reverse=True
    )
    rank_lookup = {name: index for index, name in enumerate(ranking, start=1)}

    serialized_backend_results = {
        name: _serialize_backend_result(result, rank_lookup.get(name))
        for name, result in backend_results.items()
    }
    detected_backends = [
        name for name, inventory in hardware_inventory.items() if inventory.get("probe_available")
    ]
    emit_status(
        "method.spatial_convolution.complete",
        status="running",
        method="spatial_convolution",
        detected_backends=detected_backends,
        ranking=ranking,
    )

    return {
        "schema_version": 2,
        "method": "Conv2D",
        "generated_at_unix": time.time(),
        "role": args.role,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "workload": {
            "autotune": _spec_to_dict(spec),
            "measurement": _spec_to_dict(measurement_spec),
            "autotune_repeats": DEFAULT_AUTOTUNE_REPEATS,
            "measurement_repeats": _final_measurement_repeats(
                measurement_dataset_variant == WORKLOAD_MODE_LARGE
            ),
            "workload_mode": workload_mode,
            "autotune_dataset_variant": autotune_dataset_variant,
            "measurement_dataset_variant": measurement_dataset_variant,
            "full_runtime_measurement": workload_mode == WORKLOAD_MODE_FULL,
            "output_channel_batch_candidates": cuda_backend._candidate_output_channel_batches(spec),
            "output_channel_batch_scale": cuda_backend.SPATIAL_CUDA_OUTPUT_CHANNEL_BATCH_SCALE,
            "cooldown_ms": cuda_backend.SPATIAL_CUDA_COOLDOWN_MS,
        },
        "hardware_inventory": hardware_inventory,
        "detected_backends": detected_backends,
        "usable_backends": ranking,
        "best_backend": ranking[0] if ranking else None,
        "ranking": ranking,
        "backend_results": serialized_backend_results,
    }

def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the spatial benchmark runner.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code ``0`` on success.
    """
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)

    # 写入并打印结果
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBenchmark Report Summary:")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
