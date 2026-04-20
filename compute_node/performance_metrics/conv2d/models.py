"""Shared data objects for the conv2d benchmark workspace.

Use these dataclasses when conv2d benchmark code needs stable, named
structures for workload specs, dataset layouts, per-trial metrics, and
per-backend benchmark summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AUTOTUNE_REPEATS = 3
DEFAULT_MEASUREMENT_REPEATS = 20

@dataclass(slots=True)
class BenchmarkSpec:
    """Describe one fixed Conv2D benchmark workload and its scoring window."""

    name: str
    h: int
    w: int
    c_in: int
    c_out: int
    k: int
    pad: int
    ideal_seconds: float
    zero_score_seconds: float
    stride: int = 1

    @property
    def input_bytes(self) -> int:
        """Return the byte size of the input tensor.

        Returns:
            ``H * W * C_in * 4`` bytes for float32 input data.
        """
        return self.h * self.w * self.c_in * 4

    @property
    def weight_bytes(self) -> int:
        """Return the byte size of the weight tensor.

        Returns:
            ``K * K * C_in * C_out * 4`` bytes for float32 weights.
        """
        return self.k * self.k * self.c_in * self.c_out * 4

    @property
    def output_h(self) -> int:
        """Return the output tensor height implied by the convolution shape."""
        return (self.h + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_w(self) -> int:
        """Return the output tensor width implied by the convolution shape."""
        return (self.w + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_bytes(self) -> int:
        """Return the byte size of the output tensor.

        Returns:
            ``output_h * output_w * c_out * 4`` bytes for float32 output data.
        """
        return self.output_h * self.output_w * self.c_out * 4

    @property
    def flops_per_run(self) -> int:
        """Estimate the total floating-point work for one convolution run.

        Returns:
            The multiply-add FLOP count for the resolved output tensor.
        """
        return 2 * self.output_h * self.output_w * self.c_out * self.c_in * self.k * self.k


@dataclass(slots=True)
class DatasetLayout:
    """File locations for one generated conv2d benchmark dataset."""
    root_dir: Path
    input_path: Path
    weight_path: Path
    meta_path: Path

@dataclass(slots=True)
class TrialRecord:
    """Store the best measured metrics returned by one backend trial."""

    backend: str
    config: dict[str, Any]
    wall_clock_latency_seconds: float
    effective_gflops: float
    checksum: str
    score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trial record for JSON output.

        Returns:
            A JSON-friendly dictionary representation of this trial.
        """
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
    """Store the overall outcome for one backend such as CPU or CUDA."""

    backend: str
    available: bool
    selected_config: dict[str, Any] | None
    autotune_trial: TrialRecord | None
    best_trial: TrialRecord | None
    trials: list[TrialRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Detailed per-trial timing / bandwidth breakdown emitted by the native runner
    # in --verbose mode. Benchmark-analysis only; runtime code paths ignore this.
    raw_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the backend summary for JSON output.

        Returns:
            A JSON-friendly dictionary representation of this backend result.
        """
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
