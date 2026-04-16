"""Top-level multi-method benchmark entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime_environment import relaunch_with_project_python_if_needed
from setup import active_python_path
from app.constants import DX12_BACKEND_DISABLED_REASON, METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from compute_node.performance_metrics.benchmark_status import (
    configure_status_environment,
    emit_status,
    mark_benchmark_failed,
    mark_benchmark_finished,
    mark_benchmark_started,
    resolve_status_paths,
)
from compute_node.performance_metrics.device_overview import collect_device_overview
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import (
    DATASET_DIR as FMVM_DATASET_DIR,
    RESULT_PATH as FMVM_RESULT_PATH,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication import runner as fmvm_runner
from compute_node.performance_metrics.result_format import build_report as build_normalized_report
from compute_node.performance_metrics.result_format import normalize_method_report
from compute_node.performance_metrics.spatial_convolution.config import (
    DATASET_DIR as SPATIAL_DATASET_DIR,
    RAW_BENCHMARK_PATH as SPATIAL_CONV_BENCHMARK_PATH,
    RESULT_PATH as SPATIAL_RESULT_PATH,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT_DIR / "result.json"
DEFAULT_DATASET_DIR = fmvm_runner.DEFAULT_DATASET_DIR
METHOD_RESULT_PATHS = {
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION: FMVM_RESULT_PATH,
    METHOD_SPATIAL_CONVOLUTION: SPATIAL_RESULT_PATH,
}
METHOD_DATASET_DIRS = {
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION: FMVM_DATASET_DIR,
    METHOD_SPATIAL_CONVOLUTION: SPATIAL_DATASET_DIR,
}


def build_parser() -> argparse.ArgumentParser:
    """Describe the CLI surface for one or more compute methods."""

    parser = argparse.ArgumentParser(description="Benchmark superweb-cluster compute methods.")
    parser.add_argument(
        "--method",
        choices=(
            METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
            METHOD_SPATIAL_CONVOLUTION,
            "all",
        ),
        default="all",
        help="Which method benchmark to run. Default: run both methods in sequence.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend to run. Repeatable. Applies to the selected method.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory override for FMVM. Spatial-convolution uses its own method-local dataset workspace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file that receives the benchmark report.",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=None,
        help="Crash-survivable JSON snapshot describing the last active benchmark step.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="JSONL trace file that appends every benchmark step as it runs.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=240.0,
        help="FMVM benchmark time budget in seconds. Spatial-convolution keeps its native benchmark flow.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force backend executables to rebuild instead of reusing cached binaries.",
    )
    parser.add_argument(
        "--accumulation-precision",
        choices=("fp32", "fp64_accumulate"),
        default="fp32",
        help="FMVM numeric accumulation mode. Spatial-convolution keeps its native runner behavior.",
    )
    parser.add_argument("--rows", type=int, help="Optional FMVM matrix row-count override.")
    parser.add_argument("--cols", type=int, help="Optional FMVM matrix column-count override.")
    parser.add_argument("--role", choices=("compute", "main"), default="compute", help="Spatial-convolution dataset role.")
    parser.add_argument("--h", type=int, help="Optional Conv2D test height override.")
    parser.add_argument("--w", type=int, help="Optional Conv2D test width override.")
    parser.add_argument("--cin", type=int, help="Optional Conv2D input-channel override.")
    parser.add_argument("--cout", type=int, help="Optional Conv2D output-channel override.")
    parser.add_argument("--k", type=int, help="Optional Conv2D kernel-size override.")
    parser.add_argument("--pad", type=int, help="Optional Conv2D padding override.")
    parser.add_argument("--stride", type=int, help="Optional Conv2D stride override.")
    return parser


def _selected_methods(method_arg: str) -> list[str]:
    if method_arg == "all":
        return [METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION]
    return [method_arg]


def _run_fmvm_benchmark(args: argparse.Namespace) -> dict[str, object]:
    emit_status(
        "method.fmvm.dispatch",
        status="running",
        method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        rows=args.rows,
        cols=args.cols,
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
    )
    fmvm_args = argparse.Namespace(
        backend=args.backend,
        dataset_dir=args.dataset_dir,
        output=args.output,
        time_budget=args.time_budget,
        rebuild=bool(getattr(args, "rebuild", False)),
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
        rows=args.rows,
        cols=args.cols,
    )
    previous_default_dataset_dir = fmvm_runner.DEFAULT_DATASET_DIR
    try:
        fmvm_runner.DEFAULT_DATASET_DIR = DEFAULT_DATASET_DIR
        return fmvm_runner.run_benchmark(fmvm_args)
    finally:
        fmvm_runner.DEFAULT_DATASET_DIR = previous_default_dataset_dir


def _run_spatial_convolution_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not SPATIAL_CONV_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"missing spatial_convolution benchmark entrypoint: {SPATIAL_CONV_BENCHMARK_PATH}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="superweb-spatial-benchmark-") as temp_dir:
        temp_output_path = Path(temp_dir) / "spatial_convolution_result.json"
        command = [
            str(active_python_path()),
            str(SPATIAL_CONV_BENCHMARK_PATH),
            "--output",
            str(temp_output_path),
            "--role",
            args.role,
            "--dataset-dir",
            str(METHOD_DATASET_DIRS[METHOD_SPATIAL_CONVOLUTION]),
        ]
        if args.backend:
            for backend in args.backend:
                command.extend(["--backend", backend])
        if args.rebuild:
            command.append("--rebuild")
        for field, cli_flag in (
            ("h", "--h"),
            ("w", "--w"),
            ("cin", "--cin"),
            ("cout", "--cout"),
            ("k", "--k"),
            ("pad", "--pad"),
            ("stride", "--stride"),
        ):
            value = getattr(args, field, None)
            if value is not None:
                command.extend([cli_flag, str(value)])

        emit_status(
            "method.spatial_convolution.subprocess.start",
            status="running",
            method=METHOD_SPATIAL_CONVOLUTION,
            command=command,
            cwd=str(PROJECT_ROOT),
            dataset_dir=str(METHOD_DATASET_DIRS[METHOD_SPATIAL_CONVOLUTION]),
            role=args.role,
        )
        try:
            subprocess.run(
                command,
                check=True,
                cwd=PROJECT_ROOT,
                timeout=3600.0,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            error_text = stderr or stdout or str(exc)
            if DX12_BACKEND_DISABLED_REASON in error_text:
                error_text = DX12_BACKEND_DISABLED_REASON
            emit_status(
                "method.spatial_convolution.subprocess.error",
                status="failed",
                method=METHOD_SPATIAL_CONVOLUTION,
                error=error_text,
                returncode=exc.returncode,
            )
            raise RuntimeError(error_text) from exc
        payload = json.loads(temp_output_path.read_text(encoding="utf-8"))
        payload.setdefault("benchmark_elapsed_seconds", time.perf_counter() - started)
        emit_status(
            "method.spatial_convolution.subprocess.complete",
            status="running",
            method=METHOD_SPATIAL_CONVOLUTION,
            elapsed_seconds=payload.get("benchmark_elapsed_seconds"),
            usable_backends=payload.get("usable_backends", []),
            ranking=payload.get("ranking", []),
        )
        return payload


def _method_specific_output_path(method_name: str) -> Path:
    return METHOD_RESULT_PATHS[method_name]


def _dataset_root_for_method(method_name: str, raw_report: dict[str, object]) -> str | None:
    if method_name == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
        dataset_root = str((raw_report.get("dataset") or {}).get("root_dir") or "")
        if not dataset_root:
            return None
        resolved = (ROOT_DIR / dataset_root).resolve()
        try:
            return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return dataset_root.replace("\\", "/")
    return str(METHOD_DATASET_DIRS[method_name].relative_to(PROJECT_ROOT)).replace("\\", "/")


def _normalize_results(
    raw_method_reports: dict[str, dict[str, object]],
    *,
    total_elapsed: float,
) -> dict[str, object]:
    device_overview = collect_device_overview()
    normalized_methods = {
        method_name: normalize_method_report(
            method_name=method_name,
            raw_method=raw_report,
            dataset_root=_dataset_root_for_method(method_name, raw_report),
            device_overview=device_overview,
        )
        for method_name, raw_report in raw_method_reports.items()
    }
    return build_normalized_report(
        method_reports=normalized_methods,
        device_overview=device_overview,
        total_elapsed=total_elapsed,
    )


def _write_method_reports(normalized_report: dict[str, object]) -> None:
    methods = normalized_report.get("methods", {})
    if not isinstance(methods, dict):
        return

    for method_name, method_report in methods.items():
        target_path = _method_specific_output_path(str(method_name))
        payload = {
            "schema_version": normalized_report.get("schema_version"),
            "generated_at_unix": normalized_report.get("generated_at_unix"),
            "benchmark_elapsed_seconds": normalized_report.get("benchmark_elapsed_seconds"),
            "device_overview": normalized_report.get("device_overview"),
            "methods": {
                str(method_name): method_report,
            },
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run one or more method benchmarks and return a JSON-serializable report."""

    methods = _selected_methods(getattr(args, "method", METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION))
    started = time.perf_counter()
    raw_method_reports: dict[str, dict[str, object]] = {}
    total_steps = len(methods) + 2
    for method_name in methods:
        method_index = len(raw_method_reports) + 1
        emit_status(
            "method.start",
            status="running",
            method=method_name,
            step_index=method_index,
            total_steps=total_steps,
        )
        print(
            f"[benchmark] Step {method_index}/{total_steps}: running {method_name} benchmark...",
            flush=True,
        )
        if method_name == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
            raw_method_reports[method_name] = _run_fmvm_benchmark(args)
        elif method_name == METHOD_SPATIAL_CONVOLUTION:
            raw_method_reports[method_name] = _run_spatial_convolution_benchmark(args)
        else:
            raise ValueError(f"unsupported benchmark method: {method_name}")
        elapsed = time.perf_counter() - started
        print(
            f"[benchmark] Completed {method_name} benchmark in {elapsed:.2f}s.",
            flush=True,
        )
        emit_status(
            "method.complete",
            status="running",
            method=method_name,
            step_index=method_index,
            total_steps=total_steps,
            elapsed_seconds=elapsed,
            usable_backends=(raw_method_reports[method_name] or {}).get("usable_backends", []),
            ranking=(raw_method_reports[method_name] or {}).get("ranking", []),
        )

    print(
        f"[benchmark] Step {len(methods) + 1}/{total_steps}: normalizing combined report...",
        flush=True,
    )
    emit_status(
        "benchmark.normalize.start",
        status="running",
        step_index=len(methods) + 1,
        total_steps=total_steps,
        methods=methods,
    )
    normalized_report = _normalize_results(
        raw_method_reports,
        total_elapsed=time.perf_counter() - started,
    )
    print(
        f"[benchmark] Step {len(methods) + 2}/{total_steps}: writing benchmark reports...",
        flush=True,
    )
    emit_status(
        "benchmark.write_reports.start",
        status="running",
        step_index=len(methods) + 2,
        total_steps=total_steps,
        methods=list(raw_method_reports),
    )
    _write_method_reports(normalized_report)
    emit_status(
        "benchmark.write_reports.complete",
        status="running",
        methods=list(raw_method_reports),
        ranking_summary={
            method_name: ((normalized_report.get("methods") or {}).get(method_name) or {}).get("ranking", [])
            for method_name in raw_method_reports
        },
    )
    return normalized_report


def main(argv: list[str] | None = None) -> int:
    relaunch_result = relaunch_with_project_python_if_needed(
        argv,
        script_path=Path(__file__),
        cwd=PROJECT_ROOT,
    )
    if relaunch_result is not None:
        return relaunch_result

    args = build_parser().parse_args(argv)
    status_path, trace_path = resolve_status_paths(
        output_path=args.output,
        status_path=args.status_output,
        trace_path=args.trace_output,
    )
    configure_status_environment(status_path=status_path, trace_path=trace_path)
    methods = _selected_methods(getattr(args, "method", METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION))
    mark_benchmark_started(
        argv=list(sys.argv[1:] if argv is None else argv),
        cwd=PROJECT_ROOT,
        output_path=args.output,
        methods=methods,
    )

    started = time.perf_counter()
    try:
        report = run_benchmark(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report_text = json.dumps(report, indent=2)
        args.output.write_text(report_text, encoding="utf-8")
        print(report_text, flush=True)
        mark_benchmark_finished(
            output_path=args.output,
            methods_completed=methods,
            elapsed_seconds=time.perf_counter() - started,
        )
        return 0
    except Exception as exc:
        mark_benchmark_failed(
            output_path=args.output,
            error=str(exc),
            methods_started=methods,
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
