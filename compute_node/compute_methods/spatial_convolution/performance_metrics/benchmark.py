"""Top-level benchmark entry point with hardware ranking."""

from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.compute_methods.spatial_convolution.performance_metrics.backends import (
    build_backends,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.dataset import (
    build_dataset_layout,
    dataset_is_generated,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.workloads import (
    build_benchmark_spec,
    get_runtime_spec,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark backends.")
    parser.add_argument("--backend", action="append", default=None)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT_DIR.parent / "dataset" / "generated")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "result.json")
    parser.add_argument("--role", choices=("compute", "main"), default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--cin", type=int)
    parser.add_argument("--cout", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pad", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--rebuild", action="store_true")
    return parser

def _generate_if_needed(dataset_dir: Path, spec, role: str, *, require_runtime_measurement: bool) -> None:
    """Generate datasets only when the current benchmark flow actually needs them."""
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")
    runtime_spec = get_runtime_spec()

    has_test = dataset_is_generated(test_layout, spec, skip_weight=False)
    has_runtime = dataset_is_generated(
        runtime_layout,
        runtime_spec,
        skip_weight=not require_runtime_measurement,
    )

    if has_test and (not require_runtime_measurement or has_runtime):
        return

    script = ROOT_DIR.parent / "dataset" / "generate.py"
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
    if require_runtime_measurement:
        cmd.append("--include-runtime-weight")
    else:
        cmd.append("--skip-runtime")
    emit_status(
        "method.spatial_convolution.dataset.generate.start",
        status="running",
        method="spatial_convolution",
        dataset_dir=str(dataset_dir),
        role=role,
        require_runtime_measurement=require_runtime_measurement,
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
        require_runtime_measurement=require_runtime_measurement,
    )


def _serialize_backend_result(result: dict, rank: int | None) -> dict:
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
    return any(
        getattr(args, field) is not None
        for field in ("h", "w", "cin", "cout", "k", "pad", "stride")
    )


def _spec_to_dict(spec) -> dict:
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
    return 1 if use_runtime_measurement else DEFAULT_MEASUREMENT_REPEATS


def _backend_diagnostic_context(backend, spec: BenchmarkSpec) -> dict[str, object] | None:
    builder = getattr(backend, "diagnostic_context", None)
    if not callable(builder):
        return None
    try:
        payload = builder(spec)
    except TypeError:
        payload = builder()
    return payload if isinstance(payload, dict) else None

def run_benchmark(args: argparse.Namespace) -> dict:
    emit_status(
        "method.spatial_convolution.start",
        status="running",
        method="spatial_convolution",
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        role=args.role,
        h=args.h,
        w=args.w,
        cin=args.cin,
        cout=args.cout,
        k=args.k,
        pad=args.pad,
        stride=args.stride,
    )
    spec = build_benchmark_spec(
        h=args.h,
        w=args.w,
        c_in=args.cin,
        c_out=args.cout,
        k=args.k,
        pad=args.pad,
        stride=args.stride,
    )
    use_runtime_measurement = not _has_custom_workload(args)
    measurement_spec = get_runtime_spec() if use_runtime_measurement else spec
    dataset_dir = Path(args.dataset_dir).resolve()

    # 1. 确保 benchmark 实际需要的数据存在
    _generate_if_needed(
        dataset_dir,
        spec,
        args.role,
        require_runtime_measurement=use_runtime_measurement,
    )

    # 2. 探测并运行硬件测试
    backends = build_backends(args.backend)
    backend_results = {}
    hardware_inventory = {}

    print(f"\n=== Benchmarking Backends for {args.role.upper()} NODE ===")
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    measurement_layout = (
        build_dataset_layout(dataset_dir, prefix="runtime_")
        if use_runtime_measurement
        else test_layout
    )

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
                "autotune_input": str(test_layout.input_path),
                "autotune_weight": str(test_layout.weight_path),
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
            test_layout,
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
            "measurement_repeats": _final_measurement_repeats(use_runtime_measurement),
            "full_runtime_measurement": use_runtime_measurement,
        },
        "hardware_inventory": hardware_inventory,
        "detected_backends": detected_backends,
        "usable_backends": ranking,
        "best_backend": ranking[0] if ranking else None,
        "ranking": ranking,
        "backend_results": serialized_backend_results
    }

def main(argv: list[str] | None = None) -> int:
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
