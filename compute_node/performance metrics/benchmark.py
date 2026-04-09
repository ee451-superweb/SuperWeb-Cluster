"""Top-level benchmark entry point.

This file is the conductor for the whole benchmark flow:

1. decide which fixed matrix/vector shape we want
2. check whether `A.bin` and `x.bin` already exist
3. if they do not exist, call `generate.py`
4. detect which backend executables we should run
5. keep the best result for each backend
6. rank those backend-best results and write `result.json`

This is the main place to read if you want to understand the overall control
flow. The lower-level details live in:

- `fmvm_dataset.py`: input-file generation and file layout
- `backends/`: one adapter per hardware family
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from backends import build_backends
from fmvm_dataset import build_dataset_layout, dataset_is_generated
from path_utils import to_relative_cli_path, to_relative_string
from workloads import build_benchmark_spec

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT_DIR.parent / "input matrix" / "generated"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "result.json"
GENERATE_SCRIPT_PATH = ROOT_DIR.parent / "input matrix" / "generate.py"


def build_parser() -> argparse.ArgumentParser:
    """Describe the small CLI surface for the benchmark runner."""

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
    parser.add_argument("--rows", type=int, help="Optional test-only override for the matrix row count.")
    parser.add_argument("--cols", type=int, help="Optional test-only override for the matrix column count.")
    return parser


def _resolve_dataset_dir(args: argparse.Namespace, spec) -> Path:
    """Choose where this benchmark run should read or create `A.bin` and `x.bin`.

    The default production dataset lives at `compute_node/input matrix/generated`.
    For tiny test-only shape overrides, reusing that directory is risky because
    it would overwrite the production-sized dataset with a miniature one.

    To keep the normal dataset stable, any `--rows/--cols` override that still
    points at the default directory is transparently redirected into a cached
    override-specific subdirectory.
    """

    requested_dir = Path(args.dataset_dir)
    if (args.rows is None and args.cols is None) or requested_dir != DEFAULT_DATASET_DIR:
        return requested_dir

    return requested_dir / "overrides" / f"{spec.rows}x{spec.cols}"


def _generate_dataset_if_missing(dataset_dir: Path, rows: int, cols: int) -> bool:
    """Call `generate.py` when the generated input files do not exist yet.

    Returning `True` means the benchmark had to create the dataset first.
    Returning `False` means the dataset was already present.
    """

    spec = build_benchmark_spec(rows=rows, cols=cols)
    layout = build_dataset_layout(dataset_dir)
    if dataset_is_generated(layout, spec):
        return False

    command = [
        sys.executable,
        to_relative_cli_path(GENERATE_SCRIPT_PATH, start=ROOT_DIR),
        "--output-dir",
        to_relative_cli_path(dataset_dir, start=ROOT_DIR),
        "--rows",
        str(rows),
        "--cols",
        str(cols),
    ]
    subprocess.run(command, check=True, cwd=ROOT_DIR)

    if not dataset_is_generated(layout, spec):
        raise RuntimeError("generate.py completed but the generated dataset is still incomplete")
    return True


def _trial_sort_key(result) -> tuple[float, float]:
    """Order backend best-trials from strongest to weakest.

    Higher score should rank earlier. When two backends reach the same score,
    the lower measured latency wins the tie.
    """

    if result.best_trial is None:
        return (float("-inf"), float("inf"))
    return (result.best_trial.score, -result.best_trial.wall_clock_latency_seconds)


def _serialize_backend_result(result, rank_lookup: dict[str, int]) -> dict[str, object]:
    """Convert one backend summary into the JSON layout consumed by result.json."""

    best_trial = result.best_trial
    best_config = None
    best_result = None
    trial_notes: list[str] = []
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
        "best_result": best_result,
        "notes": list(result.notes),
        "trial_notes": trial_notes,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run the benchmark and return the JSON-serializable report."""

    spec = build_benchmark_spec(rows=args.rows, cols=args.cols)
    dataset_dir = _resolve_dataset_dir(args, spec)
    dataset_was_generated = _generate_dataset_if_missing(dataset_dir, spec.rows, spec.cols)
    dataset = build_dataset_layout(dataset_dir)
    backends = build_backends(args.backend)

    total_started = time.perf_counter()
    backend_results = []
    for index, backend in enumerate(backends):
        elapsed = time.perf_counter() - total_started
        remaining = max(args.time_budget - elapsed, 1.0)
        per_backend_budget = remaining / max(len(backends) - index, 1)
        backend_results.append(
            backend.run(
                spec,
                dataset,
                time_budget_seconds=per_backend_budget,
            )
        )

    total_elapsed = time.perf_counter() - total_started

    ranked_backend_results = [
        result
        for result in sorted(backend_results, key=_trial_sort_key, reverse=True)
        if result.best_trial is not None
    ]
    rank_lookup = {
        result.backend: rank
        for rank, result in enumerate(ranked_backend_results, start=1)
    }
    backend_result_map = {
        result.backend: _serialize_backend_result(result, rank_lookup)
        for result in backend_results
    }
    ranking = [result.backend for result in ranked_backend_results]
    best_backend = ranking[0] if ranking else None

    summary: dict[str, object] = {
        "method": "fixed_matrix_vector_multiplication",
        "generated_at_unix": time.time(),
        "benchmark_elapsed_seconds": total_elapsed,
        "dataset": {
            "root_dir": to_relative_string(dataset.root_dir, start=ROOT_DIR),
            "matrix_path": to_relative_string(dataset.matrix_path, start=ROOT_DIR),
            "vector_path": to_relative_string(dataset.vector_path, start=ROOT_DIR),
            "rows": spec.rows,
            "cols": spec.cols,
            "matrix_bytes": spec.matrix_bytes,
            "vector_bytes": spec.vector_bytes,
            "dataset_was_generated": dataset_was_generated,
        },
        "backends_considered": [backend.name for backend in backends],
        "best_backend": best_backend,
        "ranking": ranking,
        "backend_results": backend_result_map,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper around `run_benchmark()`."""

    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
