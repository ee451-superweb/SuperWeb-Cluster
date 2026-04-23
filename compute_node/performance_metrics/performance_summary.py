"""Turn benchmark reports into runtime-ready local processor inventories.

Use this module when the compute node needs to convert benchmark output into
the abstract performance summaries used for registration and task execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.constants import METHOD_GEMV
from core.types import ComputeHardwarePerformance, ComputePerformanceSummary, MethodPerformanceSummary

DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "result.json"
WEAK_PROCESSOR_THRESHOLD = 0.5
DISABLED_RUNTIME_BACKENDS = frozenset({"dx12"})


@dataclass(frozen=True, slots=True)
class RuntimeProcessorProfile:
    """Describe one locally executable processor and its best benchmark config."""

    hardware_type: str
    effective_gflops: float
    rank: int
    best_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeProcessorInventory:
    """Store the filtered local processor inventory for one compute method."""

    processors: tuple[RuntimeProcessorProfile, ...]

    @property
    def total_effective_gflops(self) -> float:
        """Return the combined effective performance of all retained processors."""
        return sum(processor.effective_gflops for processor in self.processors)

    def to_method_summary(self, method: str) -> MethodPerformanceSummary:
        """Convert this runtime inventory into a registration-ready method summary.

        Args:
            method: Logical method name that owns this inventory.

        Returns:
            A ``MethodPerformanceSummary`` built from the retained processors.
        """
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
        """Build the legacy single-method registration summary for GEMV."""
        method_summary = self.to_method_summary(METHOD_GEMV)
        return ComputePerformanceSummary(
            hardware_count=method_summary.hardware_count,
            ranked_hardware=list(method_summary.ranked_hardware),
            method_summaries=[method_summary],
        )


@dataclass(frozen=True, slots=True)
class RuntimeMethodCatalog:
    """Store method-indexed inventories used during registration and execution."""

    method_inventories: dict[str, RuntimeProcessorInventory]

    def inventory_for(self, method: str) -> RuntimeProcessorInventory:
        """Return the runtime inventory for one method, or an empty fallback."""
        return self.method_inventories.get(method, RuntimeProcessorInventory(processors=()))

    def to_summary(self) -> ComputePerformanceSummary:
        """Convert the full method catalog into the advertised performance summary."""
        ordered_methods = sorted(
            self.method_inventories,
            key=lambda name: (0 if name == METHOD_GEMV else 1, name),
        )
        method_summaries = [
            self.method_inventories[method].to_method_summary(method)
            for method in ordered_methods
        ]

        legacy_view = next(
            (
                summary
                for summary in method_summaries
                if summary.method == METHOD_GEMV
            ),
            method_summaries[0] if method_summaries else MethodPerformanceSummary(method=METHOD_GEMV),
        )
        return ComputePerformanceSummary(
            hardware_count=legacy_view.hardware_count,
            ranked_hardware=list(legacy_view.ranked_hardware),
            method_summaries=method_summaries,
        )


def _read_result_payload(result_path: Path | None = None) -> dict[str, Any]:
    """Read the benchmark result payload that seeds runtime processor inventory."""
    resolved_path = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _iter_ranked_backend_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Yield ranked backend entries that are usable for runtime execution.

    Args:
        payload: Method or combined benchmark-report payload.

    Returns:
        Ranked backend entries with normalized runtime metadata.
    """
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
    """Drop processors whose performance is too weak compared with the group.

    Args:
        processors: Discovered runtime processor profiles.

    Returns:
        The retained processor profiles re-ranked by effective GFLOPS.
    """
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


def _build_inventory_from_method_payload(
    payload: dict[str, Any],
    *,
    pinned_backend: str | None = None,
) -> RuntimeProcessorInventory:
    """Build a runtime inventory from one method section of the benchmark report.

    Args:
        payload: Method-local benchmark-report payload.
        pinned_backend: When set, retain only the ranked entry matching this
            backend name so the runtime advertises exactly one backend's capacity.
    """
    ranked_entries = _iter_ranked_backend_entries(payload)
    if pinned_backend is not None:
        ranked_entries = [
            (name, entry) for name, entry in ranked_entries if name == pinned_backend
        ]
    discovered_processors = [
        RuntimeProcessorProfile(
            hardware_type=str(entry["hardware_type"]),
            effective_gflops=float(entry["effective_gflops"]),
            rank=int(entry["rank"]),
            best_config=dict(entry["best_config"]),
        )
        for _, entry in ranked_entries
    ]
    filtered_processors = _filter_weak_processors(discovered_processors)
    return RuntimeProcessorInventory(processors=tuple(filtered_processors))


def _iter_method_payloads(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Extract per-method payloads from combined or method-local benchmark reports."""
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

    method = str(payload.get("method") or METHOD_GEMV)
    return [(method, payload)]


def load_runtime_method_catalog(
    result_path: Path | None = None,
    *,
    pinned_backend: str | None = None,
) -> RuntimeMethodCatalog:
    """Load the method-indexed runtime inventory catalog from benchmark output.

    Args:
        result_path: Optional benchmark-result path override.
        pinned_backend: When set, restrict every method's inventory to only the
            named backend so dual-purpose peers advertise one backend's capacity.

    Returns:
        The runtime method catalog derived from the report.
    """
    payload = _read_result_payload(result_path)
    inventories = {
        method: _build_inventory_from_method_payload(
            method_payload, pinned_backend=pinned_backend
        )
        for method, method_payload in _iter_method_payloads(payload)
    }
    return RuntimeMethodCatalog(method_inventories=inventories)


def load_runtime_processor_inventory(
    result_path: Path | None = None,
    *,
    method: str = METHOD_GEMV,
    pinned_backend: str | None = None,
) -> RuntimeProcessorInventory:
    """Load local processor configs for one method and filter out weak processors.

    Args:
        result_path: Optional benchmark-result path override.
        method: Logical method name whose processors should be loaded.
        pinned_backend: Optional backend name restricting the inventory to a
            single backend; passed through to the method catalog loader.

    Returns:
        The filtered runtime processor inventory for the selected method.
    """

    return load_runtime_method_catalog(
        result_path, pinned_backend=pinned_backend
    ).inventory_for(method)


def load_compute_performance_summary(
    result_path: Path | None = None,
    *,
    pinned_backend: str | None = None,
) -> ComputePerformanceSummary:
    """Load the filtered runtime summary that compute-node registration should advertise.

    Args:
        result_path: Optional benchmark-result path override.
        pinned_backend: Optional backend name restricting the summary to one
            backend so the main node sees just that backend's capacity.

    Returns:
        The compute-performance summary advertised during worker registration.
    """

    return load_runtime_method_catalog(
        result_path, pinned_backend=pinned_backend
    ).to_summary()
