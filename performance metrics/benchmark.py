"""Benchmark runner for the performance-matrix workspace."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backends import build_backends
from fmvm_dataset import compute_reference_output, ensure_dataset
from scoring import MAX_SCORE, scoring_formula_description
from workloads import PRESET_WORKLOADS, resolve_workload

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "input matrix" / "generated"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "result.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark fixed matrix-vector multiplication backends.")
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Backend to run: cpu, cuda, or all. Repeatable. Default: cpu + cuda.",
    )
    parser.add_argument(
        "--preset",
        default="standard",
        choices=sorted(PRESET_WORKLOADS),
        help="Workload preset chosen to stay within a short benchmark window.",
    )
    parser.add_argument("--rows", type=int, help="Override workload rows.")
    parser.add_argument("--cols", type=int, help="Override workload cols.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where generated benchmark datasets live.",
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
        "--list-presets",
        action="store_true",
        help="Print preset dimensions and scoring windows, then exit.",
    )
    return parser


def _list_presets() -> list[dict[str, object]]:
    rows = []
    for preset in sorted(PRESET_WORKLOADS):
        workload = resolve_workload(preset)
        rows.append(
            {
                "preset": preset,
                "rows": workload.rows,
                "cols": workload.cols,
                "ideal_seconds": workload.ideal_seconds,
                "zero_score_seconds": workload.zero_score_seconds,
                "flops_per_run": workload.flops_per_run,
            }
        )
    return rows


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    backends = build_backends(args.backend or ["cpu", "cuda"])
    workload = resolve_workload(args.preset, rows=args.rows, cols=args.cols)

    dataset = ensure_dataset(args.dataset_dir, workload)
    reference_output = compute_reference_output(dataset, workload)

    total_started = time.perf_counter()
    backend_results = []
    for index, backend in enumerate(backends):
        elapsed = time.perf_counter() - total_started
        remaining = max(args.time_budget - elapsed, 1.0)
        per_backend_budget = remaining / max(len(backends) - index, 1)
        result = backend.run(
            workload,
            dataset,
            reference_output,
            time_budget_seconds=per_backend_budget,
        )
        backend_results.append(result)

    total_elapsed = time.perf_counter() - total_started
    best_result = None
    for result in backend_results:
        if result.best_trial is None or not result.best_trial.verified:
            continue
        if best_result is None or result.best_trial.score > best_result.best_trial.score:
            best_result = result

    summary = {
        "method": "fixed_matrix_vector_multiplication",
        "generated_at_unix": time.time(),
        "best_backend": None,
        "best_config": None,
        "best_result": None,
    }

    if best_result is None or best_result.best_trial is None:
        return summary

    trial = best_result.best_trial
    summary["best_backend"] = best_result.backend
    summary["best_config"] = dict(best_result.selected_config or trial.config)
    summary["best_result"] = {
        "elapsed_seconds": trial.elapsed_seconds,
        "throughput_gflops": trial.throughput_gflops,
        "score": trial.score,
        "verified": trial.verified,
        "max_abs_error": trial.max_abs_error,
        "max_rel_error": trial.max_rel_error,
        "benchmark_elapsed_seconds": total_elapsed,
        "scoring_formula": scoring_formula_description(),
        "max_score": MAX_SCORE,
        "notes": list(best_result.notes) + list(trial.notes),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print(json.dumps(_list_presets(), indent=2), flush=True)
        return 0

    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
