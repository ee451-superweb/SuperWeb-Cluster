"""Run the conv2d benchmark and emit the raw method report.

Use this module when the conv2d benchmark should benchmark its backends
directly, either from the top-level runner or as a standalone CLI.
"""

from __future__ import annotations
import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode, python_utf8_command

enable_utf8_mode()

from core.constants import DEFAULT_CONV2D_CUDA_COOLDOWN_MS
from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.performance_metrics.device_overview import detect_cpu_name
from compute_node.performance_metrics.conv2d.backends import (
    build_backends,
)
from compute_node.performance_metrics.conv2d.backends import cuda_backend
from compute_node.performance_metrics.conv2d.config import (
    DATASET_DIR as METHOD_DATASET_DIR,
    GENERATE_SCRIPT_PATH as METHOD_GENERATE_SCRIPT_PATH,
    RESULT_PATH as METHOD_RESULT_PATH,
)
from compute_node.performance_metrics.conv2d.dataset import (
    build_dataset_layout,
    dataset_prefix_for_size,
    dataset_is_generated,
)
from compute_node.performance_metrics.conv2d.models import (
    BenchmarkSpec,
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)
from compute_node.performance_metrics.conv2d.workloads import (
    build_benchmark_spec,
    get_large_spec,
    get_mid_spec,
    get_small_spec,
)
from compute_node.performance_metrics.workload_modes import (
    BENCHMARK_WORKLOAD_MODE_CHOICES,
    WORKLOAD_MODE_CUSTOM,
    WORKLOAD_MODE_FULL,
    WORKLOAD_MODE_LARGE,
    WORKLOAD_MODE_MID,
    WORKLOAD_MODE_MEDIUM,
    WORKLOAD_MODE_SMALL,
)
_QUOTED_HARDWARE_PATTERN = re.compile(r"'([^']+)'")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the conv2d benchmark entrypoint.

    Args:
        None.

    Returns:
        The configured ``ArgumentParser`` for the conv2d benchmark CLI.
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
        help="Benchmark only the small, mid, or large dataset. Default: small autotune plus mid final measurement. Use large explicitly only when you really want the large dataset.",
    )
    parser.add_argument("--output-channel-batch", type=int)
    parser.add_argument("--cooldown-ms", type=float)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ask each native runner to stream per-trial timing progress to stderr, "
        "and record the full per-trial breakdown in the benchmark report's raw_report section.",
    )
    return parser


def _apply_cuda_cli_overrides(args: argparse.Namespace) -> None:
    """Apply optional CLI overrides for the spatial CUDA benchmark backend.

    Use this before running the benchmark so the CUDA backend sees the CLI's
    temporary autotune and cooldown overrides for this process only.

    Args:
        args: Parsed conv2d benchmark CLI arguments.

    Returns:
        ``None`` after module-level CUDA overrides have been updated.
    """

    cuda_backend.CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE = None
    cuda_backend.CONV2D_CUDA_COOLDOWN_MS = DEFAULT_CONV2D_CUDA_COOLDOWN_MS

    if args.output_channel_batch is not None:
        if args.output_channel_batch <= 0:
            raise ValueError("--output-channel-batch must be a positive integer")
        cuda_backend.CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE = (args.output_channel_batch,)
    if args.cooldown_ms is not None:
        if args.cooldown_ms < 0:
            raise ValueError("--cooldown-ms must be non-negative")
        cuda_backend.CONV2D_CUDA_COOLDOWN_MS = args.cooldown_ms

def _generate_if_needed(
    dataset_dir: Path,
    spec,
    role: str,
    *,
    generate_small_dataset: bool,
    generate_mid_dataset: bool,
    generate_large_dataset: bool,
) -> None:
    """Generate datasets only when the current benchmark flow actually needs them.

    Use this before backend runs so the requested small, mid, and large
    datasets exist without regenerating variants that are not needed.

    Args:
        dataset_dir: Directory that should hold the generated datasets.
        spec: Resolved autotune benchmark specification.
        role: Dataset-generation role such as ``compute`` or ``main``.
        generate_small_dataset: Whether the small dataset is required.
        generate_mid_dataset: Whether the mid dataset is required.
        generate_large_dataset: Whether the large dataset is required.

    Returns:
        ``None`` after the required dataset files exist.
    """

    small_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("small"))
    mid_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("mid"))
    large_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("large"))
    large_spec = get_large_spec()
    mid_spec = get_mid_spec()

    has_small = dataset_is_generated(small_layout, spec, skip_weight=False) if generate_small_dataset else True
    has_mid = dataset_is_generated(mid_layout, mid_spec, skip_weight=False) if generate_mid_dataset else True
    has_large = (
        dataset_is_generated(large_layout, large_spec, skip_weight=False)
        if generate_large_dataset
        else True
    )

    if has_small and has_mid and has_large:
        return

    script = METHOD_GENERATE_SCRIPT_PATH
    cmd = python_utf8_command(
        active_python_path(),
        script,
        "--output-dir",
        dataset_dir,
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
    )
    if not generate_small_dataset:
        cmd.append("--skip-small")
    if not generate_mid_dataset:
        cmd.append("--skip-mid")
    if generate_large_dataset:
        cmd.append("--include-large-weight")
    else:
        cmd.append("--skip-large")
    emit_status(
        "method.conv2d.dataset.generate.start",
        status="running",
        method="conv2d",
        dataset_dir=str(dataset_dir),
        role=role,
        generate_small_dataset=generate_small_dataset,
        generate_mid_dataset=generate_mid_dataset,
        generate_large_dataset=generate_large_dataset,
        command=cmd,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(cmd, check=True)
    emit_status(
        "method.conv2d.dataset.generate.complete",
        status="running",
        method="conv2d",
        dataset_dir=str(dataset_dir),
        role=role,
        generate_small_dataset=generate_small_dataset,
        generate_mid_dataset=generate_mid_dataset,
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
        spec: Conv2d benchmark specification object.

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


def _backend_timeout_seconds(
    autotune_dataset_variant: str,
    measurement_dataset_variant: str,
    measurement_spec: BenchmarkSpec,
) -> float:
    """Choose a backend subprocess timeout that matches the selected workload.

    Small-workload autotuning still needs enough wall-clock room for the native
    runner to finish its sweep, especially on Windows rebuild paths. Keep that
    timeout floor higher than the scoring window so the default
    small-to-mid benchmark remains practical.
    """

    timeout_seconds = float(measurement_spec.zero_score_seconds)
    if autotune_dataset_variant == WORKLOAD_MODE_SMALL and measurement_dataset_variant == WORKLOAD_MODE_SMALL:
        return max(timeout_seconds, 300.0)
    if autotune_dataset_variant == WORKLOAD_MODE_SMALL and measurement_dataset_variant in {
        WORKLOAD_MODE_MID,
        WORKLOAD_MODE_MEDIUM,
    }:
        return max(timeout_seconds, 360.0)
    # Autotune on mid/large tensors runs one long native sweep; zero_score_seconds is too small as a wall-clock cap.
    if autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        return max(timeout_seconds, 3600.0)
    if autotune_dataset_variant == WORKLOAD_MODE_LARGE:
        return max(timeout_seconds, 7200.0)
    return timeout_seconds


def _resolve_workload_plan(args: argparse.Namespace) -> tuple[str, str, str, BenchmarkSpec, BenchmarkSpec]:
    """Resolve autotune and measurement plans from CLI workload choices.

    Use this before dataset generation and backend runs so the benchmark knows
    which spec and dataset variant belong to each phase.

    Args:
        args: Parsed conv2d benchmark CLI arguments.

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

    requested_mode = getattr(args, "workload_mode", WORKLOAD_MODE_FULL) or WORKLOAD_MODE_FULL
    if requested_mode == WORKLOAD_MODE_SMALL:
        small_spec = get_small_spec()
        return (
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            WORKLOAD_MODE_SMALL,
            small_spec,
            small_spec,
        )
    if requested_mode in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        mid_spec = get_mid_spec()
        return (
            WORKLOAD_MODE_MID,
            WORKLOAD_MODE_MID,
            WORKLOAD_MODE_MID,
            mid_spec,
            mid_spec,
        )
    if requested_mode == WORKLOAD_MODE_LARGE:
        large_spec = get_large_spec()
        return (
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            WORKLOAD_MODE_LARGE,
            large_spec,
            large_spec,
        )

    small_spec = get_small_spec()
    mid_spec = get_mid_spec()
    return (
        WORKLOAD_MODE_FULL,
        WORKLOAD_MODE_SMALL,
        WORKLOAD_MODE_MID,
        small_spec,
        mid_spec,
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


def _hardware_label_for_backend(
    backend_name: str,
    probe_message: str,
    diagnostic_context: dict[str, object] | None,
) -> str:
    """Resolve one short hardware label for conv2d benchmark progress output.

    Args:
        backend_name: Logical backend name such as ``cpu`` or ``cuda``.
        probe_message: Human-readable probe message for the backend.
        diagnostic_context: Optional backend-specific diagnostic payload.

    Returns:
        A short hardware label for stdout progress lines.
    """

    diagnostic_context = diagnostic_context or {}
    for key in ("device_name", "adapter_name"):
        value = diagnostic_context.get(key)
        if value:
            return str(value)
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


def _phase_label(dataset_variant: str, spec: BenchmarkSpec) -> str:
    """Return one concise human-facing conv2d phase label."""

    return (
        f"{dataset_variant} "
        f"({spec.h}x{spec.w}, {spec.c_in}->{spec.c_out}, k={spec.k}, pad={spec.pad}, stride={spec.stride})"
    )


def _measurement_phase_title(
    autotune_dataset_variant: str,
    measurement_dataset_variant: str,
) -> str:
    """Return the stdout title for the conv2d post-autotune phase."""

    if measurement_dataset_variant != autotune_dataset_variant:
        return "Full run stage"
    return "Final measurement stage"


def _format_selected_config(config: dict[str, object] | None) -> str:
    """Render one selected benchmark config as a compact stdout/log string."""

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

def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the conv2d benchmark and return the raw method report.

    Use this from the top-level benchmark runner or tests when the caller wants
    the conv2d benchmark result as a Python dictionary.

    Args:
        args: Parsed conv2d benchmark CLI arguments.

    Returns:
        The raw conv2d method report dictionary.
    """
    _apply_cuda_cli_overrides(args)
    workload_mode, autotune_dataset_variant, measurement_dataset_variant, spec, measurement_spec = _resolve_workload_plan(args)
    emit_status(
        "method.conv2d.start",
        status="running",
        method="conv2d",
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        workload_mode=getattr(args, "workload_mode", WORKLOAD_MODE_FULL) or WORKLOAD_MODE_FULL,
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
        cooldown_ms=cuda_backend.CONV2D_CUDA_COOLDOWN_MS,
    )
    dataset_dir = Path(args.dataset_dir).resolve()
    generate_small_dataset = autotune_dataset_variant != WORKLOAD_MODE_LARGE
    generate_mid_dataset = (
        autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
        or measurement_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}
    )
    generate_large_dataset = measurement_dataset_variant == WORKLOAD_MODE_LARGE

    # 1. 确保 benchmark 实际需要的数据存在
    _generate_if_needed(
        dataset_dir,
        spec,
        args.role,
        generate_small_dataset=generate_small_dataset,
        generate_mid_dataset=generate_mid_dataset,
        generate_large_dataset=generate_large_dataset,
    )

    # 2. 探测并运行硬件测试
    backends = build_backends(args.backend)
    backend_results = {}
    hardware_inventory = {}

    backend_time_budget_seconds = _backend_timeout_seconds(
        autotune_dataset_variant,
        measurement_dataset_variant,
        measurement_spec,
    )

    print("\n=== Benchmarking conv2d Backends ===", flush=True)
    print(
        f"    Autotune stage: {_phase_label(autotune_dataset_variant, spec)}",
        flush=True,
    )
    if bool(getattr(args, "verbose", False)):
        print(
            f"[conv2d verbose] workload_mode={workload_mode} "
            f"autotune_variant={autotune_dataset_variant} "
            f"measurement_variant={measurement_dataset_variant} "
            f"force_rebuild={bool(args.rebuild)} "
            f"backend_time_budget_seconds={backend_time_budget_seconds}",
            flush=True,
        )
    small_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("small"))
    mid_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("mid"))
    large_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("large"))
    if autotune_dataset_variant == WORKLOAD_MODE_LARGE:
        autotune_layout = large_layout
    elif autotune_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        autotune_layout = mid_layout
    else:
        autotune_layout = small_layout
    if measurement_dataset_variant == WORKLOAD_MODE_LARGE:
        measurement_layout = large_layout
    elif measurement_dataset_variant in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}:
        measurement_layout = mid_layout
    else:
        measurement_layout = autotune_layout

    for b in backends:
        diagnostic_context = _backend_diagnostic_context(b, spec)
        probe_available, probe_message = b.probe()
        hardware_name = _hardware_label_for_backend(b.name, str(probe_message), diagnostic_context)
        print(f" -> Probing {b.name} on {hardware_name}...", flush=True)
        emit_status(
            "method.conv2d.backend.probe",
            status="running",
            method="conv2d",
            backend=b.name,
            hardware_name=hardware_name,
            probe_available=bool(probe_available),
            probe_message=str(probe_message),
            diagnostic_context=diagnostic_context,
        )
        hardware_inventory[b.name] = {
            "probe_available": bool(probe_available),
            "probe_message": str(probe_message),
        }
        if probe_available:
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
                "method.conv2d.backend.autotune_selected",
                status="running",
                method="conv2d",
                backend=b.name,
                hardware_name=hardware_name,
                selected_config=selected_config,
                selected_config_text=config_text,
                measurement_phase=_measurement_phase_title(
                    autotune_dataset_variant,
                    measurement_dataset_variant,
                ),
            )
        emit_status(
            "method.conv2d.backend.start",
            status="running",
            method="conv2d",
            backend=b.name,
            hardware_name=hardware_name,
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
            diagnostic_context=diagnostic_context,
            force_rebuild=bool(args.rebuild),
        )
        result = b.run(
            spec,
            autotune_layout,
            measurement_spec=measurement_spec,
            measurement_dataset=measurement_layout,
            time_budget_seconds=backend_time_budget_seconds,
            force_rebuild=args.rebuild,
            phase_callback=_announce_phase,
            verbose=bool(getattr(args, "verbose", False)),
        )
        res = result.to_dict() if hasattr(result, 'to_dict') else result
        backend_results[b.name] = res
        emit_status(
            "method.conv2d.backend.complete",
            status="running",
            method="conv2d",
            backend=b.name,
            hardware_name=hardware_name,
            available=bool(res.get("available")),
            selected_config=res.get("selected_config") or res.get("best_config"),
            notes=res.get("notes", []),
        )

        if res.get("available"):
            best = res.get("best_trial") or res.get("best_result") or {}
            gflops = best.get("effective_gflops", 0)
            print(f"    Available on {hardware_name}. Performance: {gflops:.2f} GFLOPS", flush=True)
        else:
            notes = res.get("notes", ["No details"])
            # Probe success is stored in notes[0]; real failure reasons are appended after it.
            detail_parts = notes[1:] if len(notes) > 1 else notes
            detail = "\n      ".join(detail_parts) if detail_parts else "No details"
            print(f"    Not available on {hardware_name}: {detail}", flush=True)

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
        "method.conv2d.complete",
        status="running",
        method="conv2d",
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
            "full_runtime_measurement": measurement_dataset_variant == WORKLOAD_MODE_LARGE,
            "output_channel_batch_candidates": cuda_backend._candidate_output_channel_batches(spec),
            "cooldown_ms": cuda_backend.CONV2D_CUDA_COOLDOWN_MS,
        },
        "hardware_inventory": hardware_inventory,
        "detected_backends": detected_backends,
        "usable_backends": ranking,
        "best_backend": ranking[0] if ranking else None,
        "ranking": ranking,
        "backend_results": serialized_backend_results,
    }

def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the conv2d benchmark runner.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code ``0`` on success.
    """
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)

    # Write the full report to disk, then print only a concise stdout summary.
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
            f"[benchmark] conv2d best result: "
            f"backend={best_backend} effective_gflops={float(effective_gflops):.2f}",
            flush=True,
        )
    else:
        print("[benchmark] conv2d best result: no available backend", flush=True)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
