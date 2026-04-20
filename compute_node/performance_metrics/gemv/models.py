"""Shared data objects for the performance-metrics workspace.

These dataclasses intentionally keep the rest of the code simple:

- `BenchmarkSpec` answers "what exact matrix/vector shape are we benchmarking?"
- `DatasetLayout` answers "where do A.bin and x.bin live on disk?"
- `TrialRecord` stores the best metrics reported by one compute program
- `BackendResult` stores the overall outcome for one hardware backend

Keeping these structures in one file makes the other modules easier to read,
because they can pass around named objects instead of ad-hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AUTOTUNE_REPEATS = 3
DEFAULT_MEASUREMENT_REPEATS = 20
DEFAULT_ACCUMULATION_PRECISION = "fp32"
SUPPORTED_ACCUMULATION_PRECISIONS = ("fp32", "fp64_accumulate")


@dataclass(slots=True)
class BenchmarkSpec:
    """One fixed matrix-vector benchmark problem plus its scoring window.

    The benchmark always measures the operation:

        y = A x

    where `A` is a dense float32 matrix and `x` is a dense float32 vector.
    """

    name: str
    rows: int
    cols: int
    ideal_seconds: float
    zero_score_seconds: float
    accumulation_precision: str

    @property
    def matrix_bytes(self) -> int:
        """Return the on-disk byte size of `A.bin`."""

        return self.rows * self.cols * 4

    @property
    def vector_bytes(self) -> int:
        """Return the on-disk byte size of `x.bin`."""

        return self.cols * 4

    @property
    def flops_per_run(self) -> int:
        """Count one multiply and one add per matrix entry."""

        return 2 * self.rows * self.cols


@dataclass(slots=True)
class DatasetLayout:
    """File locations for the generated input dataset."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path


@dataclass(slots=True)
class TrialRecord:
    """The best measured result returned by one backend executable."""

    backend: str
    config: dict[str, Any]
    wall_clock_latency_seconds: float
    effective_gflops: float
    checksum: str
    score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for JSON output."""

        return {
            "backend": self.backend,
            "config": dict(self.config),
            "wall_clock_latency_seconds": self.wall_clock_latency_seconds,
            "effective_gflops": self.effective_gflops,
            "checksum": self.checksum,
            "score": self.score,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class BackendResult:
    """Aggregate outcome for one backend such as CPU or CUDA."""

    backend: str
    available: bool
    selected_config: dict[str, Any] | None
    autotune_trial: TrialRecord | None
    best_trial: TrialRecord | None
    trials: list[TrialRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the backend summary for JSON output."""

        return {
            "backend": self.backend,
            "available": self.available,
            "selected_config": None if self.selected_config is None else dict(self.selected_config),
            "autotune_trial": None if self.autotune_trial is None else self.autotune_trial.to_dict(),
            "best_trial": None if self.best_trial is None else self.best_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "notes": list(self.notes),
            "raw_report": dict(self.raw_report),
        }
