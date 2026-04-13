"""Helpers for turning benchmark `result.json` into a small runtime summary."""

from __future__ import annotations

import json
from pathlib import Path

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary

DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "performance_metrics" / "result.json"


def load_compute_performance_summary(result_path: Path | None = None) -> ComputePerformanceSummary:
    """Load the ranked backend summary that bootstrap prepared locally.

    The runtime registration only needs a compact view of the benchmark:

    - how many compute backends this node can actually use
    - which backend types they are, in benchmark rank order
    - the best effective GFLOPS reported for each backend

    Full tuning configs stay on disk in `result.json`; they are intentionally
    not sent over the runtime registration message.
    """

    resolved_path = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))

    backend_results = payload.get("backend_results")
    if not isinstance(backend_results, dict):
        raise ValueError("benchmark result is missing backend_results")

    ranking = payload.get("ranking")
    if isinstance(ranking, list):
        ranked_backend_names = [str(name) for name in ranking]
    else:
        ranked_backend_names = sorted(
            backend_results,
            key=lambda name: int((backend_results.get(name) or {}).get("rank") or 10**9),
        )

    ranked_hardware: list[ComputeHardwarePerformance] = []
    for fallback_rank, backend_name in enumerate(ranked_backend_names, start=1):
        backend_entry = backend_results.get(backend_name)
        if not isinstance(backend_entry, dict):
            continue
        if not backend_entry.get("available"):
            continue

        best_result = backend_entry.get("best_result")
        if not isinstance(best_result, dict):
            continue

        effective_gflops = float(best_result.get("effective_gflops") or 0.0)
        if effective_gflops <= 0.0:
            continue

        ranked_hardware.append(
            ComputeHardwarePerformance(
                hardware_type=backend_name,
                effective_gflops=effective_gflops,
                rank=int(backend_entry.get("rank") or fallback_rank),
            )
        )

    ranked_hardware.sort(key=lambda item: item.rank)
    return ComputePerformanceSummary(
        hardware_count=len(ranked_hardware),
        ranked_hardware=ranked_hardware,
    )
