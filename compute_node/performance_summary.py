"""Helpers for turning benchmark results into runtime-ready processor inventories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, MethodPerformanceSummary

DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "performance_metrics" / "result.json"
WEAK_PROCESSOR_THRESHOLD = 0.5
DISABLED_RUNTIME_BACKENDS = frozenset({"dx12"})


@dataclass(frozen=True, slots=True)
class RuntimeProcessorProfile:
    """One locally executable processor plus its best benchmark config."""

    hardware_type: str
    effective_gflops: float
    rank: int
    best_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeProcessorInventory:
    """Filtered local processor inventory used by one compute method."""

    processors: tuple[RuntimeProcessorProfile, ...]

    @property
    def total_effective_gflops(self) -> float:
        return sum(processor.effective_gflops for processor in self.processors)

    def to_method_summary(self, method: str) -> MethodPerformanceSummary:
        ranked_hardware = [
            ComputeHardwarePerformance(
                hardware_type=processor.hardware_type,
                effective_gflops=processor.effective_gflops,
                rank=index,
            )
            for index, processor in enumerate(self.processors, start=1)
        ]
        return MethodPerformanceSummary(
            method=method,
            hardware_count=len(ranked_hardware),
            ranked_hardware=ranked_hardware,
        )

    def to_legacy_summary(self) -> ComputePerformanceSummary:
        method_summary = self.to_method_summary(METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION)
        return ComputePerformanceSummary(
            hardware_count=method_summary.hardware_count,
            ranked_hardware=list(method_summary.ranked_hardware),
            method_summaries=[method_summary],
        )


@dataclass(frozen=True, slots=True)
class RuntimeMethodCatalog:
    """Method-indexed local processor inventories used during registration and execution."""

    method_inventories: dict[str, RuntimeProcessorInventory]

    def inventory_for(self, method: str) -> RuntimeProcessorInventory:
        return self.method_inventories.get(method, RuntimeProcessorInventory(processors=()))

    def to_summary(self) -> ComputePerformanceSummary:
        ordered_methods = sorted(
            self.method_inventories,
            key=lambda name: (0 if name == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION else 1, name),
        )
        method_summaries = [
            self.method_inventories[method].to_method_summary(method)
            for method in ordered_methods
        ]

        legacy_view = next(
            (
                summary
                for summary in method_summaries
                if summary.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
            ),
            method_summaries[0] if method_summaries else MethodPerformanceSummary(method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION),
        )
        return ComputePerformanceSummary(
            hardware_count=legacy_view.hardware_count,
            ranked_hardware=list(legacy_view.ranked_hardware),
            method_summaries=method_summaries,
        )


def _read_result_payload(result_path: Path | None = None) -> dict[str, Any]:
    resolved_path = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _iter_ranked_backend_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    backend_results = payload.get("backends")
    if not isinstance(backend_results, dict):
        backend_results = payload.get("backend_results")
    if not isinstance(backend_results, dict):
        raise ValueError("benchmark result is missing backends")

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
        if str(backend_name) in DISABLED_RUNTIME_BACKENDS:
            continue
        if not backend_entry.get("available"):
            continue

        best_result = backend_entry.get("best_result") or backend_entry.get("best_trial")
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


def _build_inventory_from_method_payload(payload: dict[str, Any]) -> RuntimeProcessorInventory:
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


def _iter_method_payloads(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    methods = payload.get("methods")
    if isinstance(methods, dict):
        entries: list[tuple[str, dict[str, Any]]] = []
        for method, method_payload in methods.items():
            if isinstance(method_payload, dict):
                entries.append((str(method), method_payload))
        return entries

    method_results = payload.get("method_results")
    if isinstance(method_results, dict):
        entries: list[tuple[str, dict[str, Any]]] = []
        for method, method_payload in method_results.items():
            if isinstance(method_payload, dict):
                entries.append((str(method), method_payload))
        return entries

    method = str(payload.get("method") or METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION)
    return [(method, payload)]


def load_runtime_method_catalog(result_path: Path | None = None) -> RuntimeMethodCatalog:
    payload = _read_result_payload(result_path)
    inventories = {
        method: _build_inventory_from_method_payload(method_payload)
        for method, method_payload in _iter_method_payloads(payload)
    }
    return RuntimeMethodCatalog(method_inventories=inventories)


def load_runtime_processor_inventory(
    result_path: Path | None = None,
    *,
    method: str = METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
) -> RuntimeProcessorInventory:
    """Load local processor configs for one method and filter out weak processors."""

    return load_runtime_method_catalog(result_path).inventory_for(method)


def load_compute_performance_summary(result_path: Path | None = None) -> ComputePerformanceSummary:
    """Load the filtered runtime summary that compute-node registration should advertise."""

    return load_runtime_method_catalog(result_path).to_summary()
