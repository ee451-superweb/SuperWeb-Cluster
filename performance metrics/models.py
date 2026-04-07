"""Shared dataclasses for the performance benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkloadSpec:
    """One benchmark workload for fixed matrix-vector multiplication."""

    name: str
    preset: str
    rows: int
    cols: int
    ideal_seconds: float
    zero_score_seconds: float
    verify_atol: float = 1e-3
    verify_rtol: float = 1e-4

    @property
    def flops_per_run(self) -> int:
        return 2 * self.rows * self.cols


@dataclass(slots=True)
class DatasetPaths:
    """Files that define one deterministic benchmark dataset."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path
    transposed_matrix_path: Path


@dataclass(slots=True)
class TrialRecord:
    """One measured trial for a backend/configuration pair."""

    backend: str
    config: dict[str, Any]
    elapsed_seconds: float
    throughput_gflops: float
    score: float
    verified: bool
    max_abs_error: float
    max_rel_error: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "config": self.config,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_gflops": self.throughput_gflops,
            "score": self.score,
            "verified": self.verified,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class BackendResult:
    """Aggregated benchmark outcome for one backend."""

    backend: str
    available: bool
    selected_config: dict[str, Any] | None
    best_trial: TrialRecord | None
    trials: list[TrialRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "selected_config": self.selected_config,
            "best_trial": None if self.best_trial is None else self.best_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "notes": list(self.notes),
        }
