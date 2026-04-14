"""Helpers for turning benchmark ``result.json`` into runtime-ready processor inventories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary

DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "performance_metrics" / "result.json"
WEAK_PROCESSOR_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class RuntimeProcessorProfile:
    """One locally executable processor plus its best benchmark config."""

    hardware_type: str
    effective_gflops: float
    rank: int
    best_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeProcessorInventory:
    """Filtered local processor inventory used by compute-node runtime execution."""

    processors: tuple[RuntimeProcessorProfile, ...]

    @property
    def total_effective_gflops(self) -> float:
        return sum(processor.effective_gflops for processor in self.processors)

    def to_summary(self) -> ComputePerformanceSummary:
        ranked_hardware = [
            ComputeHardwarePerformance(
                hardware_type=processor.hardware_type,
                effective_gflops=processor.effective_gflops,
                rank=index,
            )
            for index, processor in enumerate(self.processors, start=1)
        ]
        return ComputePerformanceSummary(
            hardware_count=len(ranked_hardware),
            ranked_hardware=ranked_hardware,
        )


def _read_result_payload(result_path: Path | None = None) -> dict[str, Any]:
    resolved_path = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _iter_ranked_backend_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
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

    ranked_entries: list[tuple[str, dict[str, Any]]] = []
    for fallback_rank, backend_name in enumerate(ranked_backend_names, start=1):
        backend_entry = backend_results.get(backend_name)
        if not isinstance(backend_entry, dict):
            continue
        if not backend_entry.get("available"):
            continue

        best_result = backend_entry.get("best_result")
        best_config = backend_entry.get("best_config")
        if not isinstance(best_result, dict) or not isinstance(best_config, dict):
            continue

        effective_gflops = float(best_result.get("effective_gflops") or 0.0)
        if effective_gflops <= 0.0:
            continue

        ranked_entries.append(
            (
                str(backend_name),
                {
                    "hardware_type": str(backend_name),
                    "effective_gflops": effective_gflops,
                    "rank": int(backend_entry.get("rank") or fallback_rank),
                    "best_config": dict(best_config),
                },
            )
        )
    return ranked_entries


def _filter_weak_processors(processors: list[RuntimeProcessorProfile]) -> list[RuntimeProcessorProfile]:
    """Drop processors whose GFLOPS stays below half of the current average."""

    retained = list(sorted(processors, key=lambda item: item.rank))
    while len(retained) > 1:
        average = sum(item.effective_gflops for item in retained) / len(retained)
        threshold = average * WEAK_PROCESSOR_THRESHOLD
        filtered = [item for item in retained if item.effective_gflops >= threshold]
        if len(filtered) == len(retained):
            break
        if not filtered:
            retained = [max(retained, key=lambda item: item.effective_gflops)]
            break
        retained = filtered

    retained.sort(key=lambda item: item.effective_gflops, reverse=True)
    return [
        RuntimeProcessorProfile(
            hardware_type=item.hardware_type,
            effective_gflops=item.effective_gflops,
            rank=index,
            best_config=dict(item.best_config),
        )
        for index, item in enumerate(retained, start=1)
    ]


def load_runtime_processor_inventory(result_path: Path | None = None) -> RuntimeProcessorInventory:
    """Load local processor configs and filter out weak processors before runtime registration."""

    payload = _read_result_payload(result_path)
    discovered_processors = [
        RuntimeProcessorProfile(
            hardware_type=str(entry["hardware_type"]),
            effective_gflops=float(entry["effective_gflops"]),
            rank=int(entry["rank"]),
            best_config=dict(entry["best_config"]),
        )
        for _, entry in _iter_ranked_backend_entries(payload)
    ]
    filtered_processors = _filter_weak_processors(discovered_processors)
    return RuntimeProcessorInventory(processors=tuple(filtered_processors))


def load_compute_performance_summary(result_path: Path | None = None) -> ComputePerformanceSummary:
    """Load the filtered runtime summary that compute-node registration should advertise."""

    return load_runtime_processor_inventory(result_path).to_summary()
