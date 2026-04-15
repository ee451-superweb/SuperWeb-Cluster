"""Shared data objects for the performance-metrics workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AUTOTUNE_REPEATS = 3
DEFAULT_MEASUREMENT_REPEATS = 20

@dataclass(slots=True)
class BenchmarkSpec:
    """One fixed Convolution (Conv2D) benchmark problem.

    The benchmark measures: Output = Conv2D(Input, Weights)
    """

    name: str
    h: int      # 图像高度
    w: int      # 图像宽度
    c_in: int   # 输入通道数
    c_out: int  # 输出通道数 (卷积核个数)
    k: int      # 卷积核大小 (例如 3 代表 3x3)
    pad: int    # 边缘填充 (Padding)
    ideal_seconds: float
    zero_score_seconds: float
    stride: int = 1

    @property
    def input_bytes(self) -> int:
        """输入特征图大小: H * W * C_in * 4 bytes"""
        return self.h * self.w * self.c_in * 4

    @property
    def weight_bytes(self) -> int:
        """卷积核权重大小: K * K * C_in * C_out * 4 bytes"""
        return self.k * self.k * self.c_in * self.c_out * 4

    @property
    def output_h(self) -> int:
        return (self.h + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_w(self) -> int:
        return (self.w + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_bytes(self) -> int:
        return self.output_h * self.output_w * self.c_out * 4

    @property
    def flops_per_run(self) -> int:
        """计算卷积所需的浮点运算次数 (FLOPs)"""
        # 每个输出像素点需要进行 K * K * C_in 次乘加运算
        return 2 * self.output_h * self.output_w * self.c_out * self.c_in * self.k * self.k


@dataclass(slots=True)
class DatasetLayout:
    """File locations for the generated input dataset."""
    root_dir: Path
    input_path: Path
    weight_path: Path
    meta_path: Path

# --- 下方 TrialRecord 和 BackendResult 代码保持不变 ---
@dataclass(slots=True)
class TrialRecord:
    backend: str
    config: dict[str, Any]
    wall_clock_latency_seconds: float
    effective_gflops: float
    checksum: str
    score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
    backend: str
    available: bool
    selected_config: dict[str, Any] | None
    autotune_trial: TrialRecord | None
    best_trial: TrialRecord | None
    trials: list[TrialRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "selected_config": None if self.selected_config is None else dict(self.selected_config),
            "autotune_trial": None if self.autotune_trial is None else self.autotune_trial.to_dict(),
            "best_trial": None if self.best_trial is None else self.best_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "notes": list(self.notes),
        }
