"""Run one or more benchmark methods and normalize their reports.

Use this module as the top-level benchmark CLI when the project should execute
GEMV, conv2d, or both in sequence and emit a unified report.
"""

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

from adapters.process import enable_utf8_mode, python_utf8_command

enable_utf8_mode()

from core.venv import relaunch_with_project_python_if_needed
from setup import active_python_path
from core.constants import DX12_BACKEND_DISABLED_REASON, METHOD_GEMM, METHOD_GEMV, METHOD_CONV2D
from compute_node.performance_metrics.benchmark_status import (
    configure_status_environment,
    emit_status,
    mark_benchmark_failed,
    mark_benchmark_finished,
    mark_benchmark_started,
    resolve_status_paths,
)
from compute_node.performance_metrics.device_overview import collect_device_overview
from compute_node.performance_metrics.gemv.config import (
    DATASET_DIR as GEMV_DATASET_DIR,
    RESULT_PATH as GEMV_RESULT_PATH,
)
from compute_node.performance_metrics.gemv import runner as gemv_runner
from compute_node.performance_metrics.result_format import build_report as build_normalized_report
from compute_node.performance_metrics.result_format import normalize_method_report
from compute_node.performance_metrics.conv2d.config import (
    DATASET_DIR as CONV2D_DATASET_DIR,
    RAW_BENCHMARK_PATH as CONV2D_BENCHMARK_PATH,
    RESULT_PATH as CONV2D_RESULT_PATH,
)
from compute_node.performance_metrics.gemm.config import (
    DATASET_DIR as GEMM_DATASET_DIR,
    RESULT_PATH as GEMM_RESULT_PATH,
)
from compute_node.performance_metrics.gemm import benchmark as gemm_benchmark
from compute_node.performance_metrics.workload_modes import (
    BENCHMARK_WORKLOAD_MODE_CHOICES,
    WORKLOAD_MODE_FULL,
    WORKLOAD_MODE_SMALL,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT_DIR / "result.json"
DEFAULT_DATASET_DIR = gemv_runner.DEFAULT_DATASET_DIR
METHOD_RESULT_PATHS = {
    METHOD_GEMV: GEMV_RESULT_PATH,
    METHOD_CONV2D: CONV2D_RESULT_PATH,
    METHOD_GEMM: GEMM_RESULT_PATH,
}
METHOD_DATASET_DIRS = {
    METHOD_GEMV: GEMV_DATASET_DIR,
    METHOD_CONV2D: CONV2D_DATASET_DIR,
    METHOD_GEMM: GEMM_DATASET_DIR,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the multi-method benchmark entrypoint.

    Use this in the top-level benchmark CLI so callers can select methods,
    workload sizes, dataset overrides, and output locations consistently.

    Args:
        None.

    Returns:
        The configured ``ArgumentParser`` for this entrypoint.
    """

    parser = argparse.ArgumentParser(description="Benchmark superweb-cluster compute methods.")
    parser.add_argument(
        "--method",
        choices=(
            METHOD_GEMV,
            METHOD_CONV2D,
            METHOD_GEMM,
            "all",
        ),
        default="all",
        help="Which method benchmark to run. Default: run every method in sequence.",
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
        help="Dataset directory override for GEMV. Conv2d uses its own method-local dataset workspace.",
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
        help="GEMV benchmark time budget in seconds. Conv2d keeps its native benchmark flow.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force backend executables to rebuild instead of reusing cached binaries.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream per-trial autotune progress from native runners and record the "
        "full per-trial breakdown in each method's raw_report section.",
    )
    parser.add_argument(
        "--accumulation-precision",
        choices=("fp32", "fp64_accumulate"),
        default="fp32",
        help="GEMV numeric accumulation mode. Conv2d keeps its native runner behavior.",
    )
    parser.add_argument("--rows", type=int, help="Optional GEMV matrix row-count override.")
    parser.add_argument("--cols", type=int, help="Optional GEMV matrix column-count override.")
    parser.add_argument(
        "--workload-mode",
        choices=BENCHMARK_WORKLOAD_MODE_CHOICES,
        default=None,
        help="Optional workload-size override. When omitted, GEMV keeps its default full flow and conv2d defaults to small autotune plus mid final measurement.",
    )
    parser.add_argument("--role", choices=("compute", "main"), default="compute", help="Conv2d dataset role.")
    parser.add_argument("--h", type=int, help="Optional Conv2D small-size height override.")
    parser.add_argument("--w", type=int, help="Optional Conv2D small-size width override.")
    parser.add_argument("--cin", type=int, help="Optional Conv2D input-channel override.")
    parser.add_argument("--cout", type=int, help="Optional Conv2D output-channel override.")
    parser.add_argument("--k", type=int, help="Optional Conv2D kernel-size override.")
    parser.add_argument("--pad", type=int, help="Optional Conv2D padding override.")
    parser.add_argument("--stride", type=int, help="Optional Conv2D stride override.")
    return parser


def _selected_methods(method_arg: str) -> list[str]:
    """Expand the CLI method selector into an ordered method list.

    Use this after parsing CLI arguments so downstream code can always iterate a
    concrete list instead of branching on the special ``all`` value.

    Args:
        method_arg: CLI ``--method`` value.

    Returns:
        The ordered list of method names to benchmark.
    """
    if method_arg == "all":
        # GEMM first: cuBLAS is vendor-tuned (not a hand-rolled kernel) and
        # runs fastest, so front-loading it gets an early usable result out
        # before the longer GEMV/conv2d autotune sweeps finish.
        return [METHOD_GEMM, METHOD_GEMV, METHOD_CONV2D]
    return [method_arg]


def _run_gemv_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the GEMV benchmark entrypoint and return its raw report.

    Use this helper from the combined benchmark flow so GEMV can keep its own
    runner while still contributing to the unified top-level report.

    Args:
        args: Parsed top-level benchmark CLI arguments.

    Returns:
        The raw JSON-like GEMV report dictionary.
    """
    workload_mode = getattr(args, "workload_mode", None) or WORKLOAD_MODE_FULL
    emit_status(
        "method.gemv.dispatch",
        status="running",
        method=METHOD_GEMV,
        requested_backends=args.backend or ["auto"],
        rebuild=bool(getattr(args, "rebuild", False)),
        workload_mode=workload_mode,
        rows=args.rows,
        cols=args.cols,
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
    )
    gemv_args = argparse.Namespace(
        backend=args.backend,
        dataset_dir=args.dataset_dir,
        output=args.output,
        time_budget=args.time_budget,
        rebuild=bool(getattr(args, "rebuild", False)),
        accumulation_precision=getattr(args, "accumulation_precision", "fp32"),
        rows=args.rows,
        cols=args.cols,
        workload_mode=workload_mode,
        verbose=bool(getattr(args, "verbose", False)),
    )
    previous_default_dataset_dir = gemv_runner.DEFAULT_DATASET_DIR
    try:
        gemv_runner.DEFAULT_DATASET_DIR = DEFAULT_DATASET_DIR
        return gemv_runner.run_benchmark(gemv_args)
    finally:
        gemv_runner.DEFAULT_DATASET_DIR = previous_default_dataset_dir


def _run_conv2d_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the conv2d benchmark as a subprocess.

    Use this helper from the combined benchmark flow so the method-specific
    conv2d benchmark can keep its own CLI and workspace conventions.

    Args:
        args: Parsed top-level benchmark CLI arguments.

    Returns:
        The raw JSON-like conv2d report dictionary.
    """
    workload_mode = getattr(args, "workload_mode", None) or WORKLOAD_MODE_FULL
    if not CONV2D_BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"missing conv2d benchmark entrypoint: {CONV2D_BENCHMARK_PATH}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="superweb-spatial-benchmark-") as temp_dir:
        temp_output_path = Path(temp_dir) / "conv2d_result.json"
        command = python_utf8_command(
            active_python_path(),
            CONV2D_BENCHMARK_PATH,
            "--output",
            temp_output_path,
            "--role",
            args.role,
            "--dataset-dir",
            METHOD_DATASET_DIRS[METHOD_CONV2D],
            "--workload-mode",
            workload_mode,
        )
        if args.backend:
            for backend in args.backend:
                command.extend(["--backend", backend])
        if args.rebuild:
            command.append("--rebuild")
        if getattr(args, "verbose", False):
            command.append("--verbose")
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
            "method.conv2d.subprocess.start",
            status="running",
            method=METHOD_CONV2D,
            command=command,
            cwd=str(PROJECT_ROOT),
            dataset_dir=str(METHOD_DATASET_DIRS[METHOD_CONV2D]),
            role=args.role,
        )
        output_lines: list[str] = []
        try:
            with subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output_lines.append(line)
                return_code = process.wait(timeout=3600.0)
            if return_code != 0:
                error_text = "".join(output_lines).strip() or f"conv2d benchmark exited with code {return_code}"
                if DX12_BACKEND_DISABLED_REASON in error_text:
                    error_text = DX12_BACKEND_DISABLED_REASON
                emit_status(
                    "method.conv2d.subprocess.error",
                    status="failed",
                    method=METHOD_CONV2D,
                    error=error_text,
                    returncode=return_code,
                )
                raise RuntimeError(error_text)
        except subprocess.TimeoutExpired as exc:
            error_text = f"conv2d benchmark timed out after {exc.timeout} seconds"
            emit_status(
                "method.conv2d.subprocess.error",
                status="failed",
                method=METHOD_CONV2D,
                error=error_text,
            )
            raise RuntimeError(error_text) from exc
        except OSError as exc:
            error_text = str(exc)
            if DX12_BACKEND_DISABLED_REASON in error_text:
                error_text = DX12_BACKEND_DISABLED_REASON
            emit_status(
                "method.conv2d.subprocess.error",
                status="failed",
                method=METHOD_CONV2D,
                error=error_text,
            )
            raise RuntimeError(error_text) from exc
        payload = json.loads(temp_output_path.read_text(encoding="utf-8"))
        payload.setdefault("benchmark_elapsed_seconds", time.perf_counter() - started)
        emit_status(
            "method.conv2d.subprocess.complete",
            status="running",
            method=METHOD_CONV2D,
            elapsed_seconds=payload.get("benchmark_elapsed_seconds"),
            usable_backends=payload.get("usable_backends", []),
            ranking=payload.get("ranking", []),
        )
        return payload


def _run_gemm_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the GEMM benchmark (CPU + optional cuBLAS) and return its raw report.

    Use this helper from the combined benchmark flow. The GEMM benchmark
    always runs a CPU pass (self-contained threaded SGEMM) and, when
    ``nvidia-smi`` is detectable on this host, also runs the cuBLAS pass and
    picks the faster of the two as ``best_backend``. We call the benchmark
    module in-process and reuse its JSON as the raw report.

    Args:
        args: Parsed top-level benchmark CLI arguments.

    Returns:
        The raw JSON-like GEMM report dictionary.
    """
    workload_mode = getattr(args, "workload_mode", None) or WORKLOAD_MODE_FULL
    size = "mid" if workload_mode not in {"small", "mid", "large"} else workload_mode
    emit_status(
        "method.gemm.dispatch",
        status="running",
        method=METHOD_GEMM,
        requested_backends=["cpu", "cuda"],
        rebuild=bool(getattr(args, "rebuild", False)),
        workload_mode=workload_mode,
        size=size,
    )
    target_output = GEMM_RESULT_PATH
    gemm_benchmark.run(
        size=size,
        iteration_count=gemm_benchmark.DEFAULT_ITERATION_COUNT,
        output=target_output,
        dataset_dir=METHOD_DATASET_DIRS[METHOD_GEMM],
        force_rebuild=bool(getattr(args, "rebuild", False)),
    )
    payload = json.loads(target_output.read_text(encoding="utf-8"))
    emit_status(
        "method.gemm.complete",
        status="running",
        method=METHOD_GEMM,
        elapsed_seconds=payload.get("benchmark_elapsed_seconds"),
        usable_backends=payload.get("usable_backends", []),
        ranking=payload.get("ranking", []),
    )
    return payload


def _method_specific_output_path(method_name: str) -> Path:
    """Return the canonical per-method result path for a method name.

    Args:
        method_name: Logical compute method name.

    Returns:
        The method-local ``result.json`` path.
    """
    return METHOD_RESULT_PATHS[method_name]


def _dataset_root_for_method(method_name: str, raw_report: dict[str, object]) -> str | None:
    """Resolve the report-friendly dataset root string for one method.

    Use this during normalization so each method report points at stable,
    project-relative dataset paths instead of machine-specific absolute paths.

    Args:
        method_name: Logical compute method name.
        raw_report: Raw method report emitted by the benchmark runner.

    Returns:
        A portable dataset-root string, or ``None`` if unavailable.
    """
    if method_name == METHOD_GEMV:
        dataset_root = str((raw_report.get("dataset") or {}).get("root_dir") or "")
        if not dataset_root:
            return None
        resolved = (ROOT_DIR / dataset_root).resolve()
        try:
            return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return dataset_root.replace("\\", "/")
    try:
        return str(METHOD_DATASET_DIRS[method_name].relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(METHOD_DATASET_DIRS[method_name]).replace("\\", "/")


def _normalize_results(
    raw_method_reports: dict[str, dict[str, object]],
    *,
    total_elapsed: float,
) -> dict[str, object]:
    """Normalize raw per-method outputs into the shared report schema.

    Use this once all selected methods have finished so the top-level output has
    a consistent structure regardless of each method's native report shape.

    Args:
        raw_method_reports: Raw reports keyed by method name.
        total_elapsed: Total wall-clock time for the combined benchmark run.

    Returns:
        The normalized benchmark report dictionary.
    """
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
    """Write one method-local report file for each normalized method result.

    Use this after normalization so each method keeps its own ``result.json`` in
    addition to the combined multi-method report.

    Args:
        normalized_report: Combined normalized benchmark report.

    Returns:
        ``None`` after all method-local reports have been written.
    """
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


def _print_benchmark_summary(normalized_report: dict[str, object]) -> None:
    """Print one concise operator-facing summary for each completed method.

    Args:
        normalized_report: Combined normalized benchmark report.

    Returns:
        ``None`` after summary lines have been written to stdout.
    """

    methods = normalized_report.get("methods", {})
    if not isinstance(methods, dict):
        return

    print("\nBenchmark Summary:", flush=True)
    for method_name, method_report in methods.items():
        if not isinstance(method_report, dict):
            continue
        best_backend = str(method_report.get("best_backend") or "")
        backends = method_report.get("backends", {})
        best_result = None
        if best_backend and isinstance(backends, dict):
            best_backend_report = backends.get(best_backend)
            if isinstance(best_backend_report, dict):
                best_result = best_backend_report.get("best_result")
        effective_gflops = None
        if isinstance(best_result, dict):
            effective_gflops = best_result.get("effective_gflops")
        if best_backend and effective_gflops is not None:
            print(
                f"[benchmark] {method_name} best result: "
                f"backend={best_backend} effective_gflops={float(effective_gflops):.2f}",
                flush=True,
            )
        else:
            print(
                f"[benchmark] {method_name} best result: no available backend",
                flush=True,
            )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the requested benchmarks and return a normalized report.

    Use this from the CLI entrypoint or tests when the caller wants the
    benchmark result as a Python dictionary instead of just a written file.

    Args:
        args: Parsed benchmark CLI arguments.

    Returns:
        The normalized multi-method benchmark report.
    """

    methods = _selected_methods(getattr(args, "method", METHOD_GEMV))
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
        if method_name == METHOD_GEMV:
            raw_method_reports[method_name] = _run_gemv_benchmark(args)
        elif method_name == METHOD_CONV2D:
            raw_method_reports[method_name] = _run_conv2d_benchmark(args)
        elif method_name == METHOD_GEMM:
            raw_method_reports[method_name] = _run_gemm_benchmark(args)
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
    """CLI entrypoint for the combined benchmark runner.

    Use this when launching the benchmark from the command line so Python
    relaunch, status logging, and final report writing all happen consistently.

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
    status_path, trace_path = resolve_status_paths(
        output_path=args.output,
        status_path=args.status_output,
        trace_path=args.trace_output,
    )
    configure_status_environment(status_path=status_path, trace_path=trace_path)
    methods = _selected_methods(getattr(args, "method", METHOD_GEMV))
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
        _print_benchmark_summary(report)
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
