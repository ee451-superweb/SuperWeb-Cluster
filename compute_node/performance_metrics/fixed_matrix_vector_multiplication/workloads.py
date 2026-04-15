"""Benchmark constants for the fixed matrix-vector workload.

This module is intentionally small. Its whole job is to answer:

- what shape should `A` and `x` have by default?
- what score window should we use when mapping latency to a linear score?

The default matrix shape below is exactly the one requested by the user:

- `A`: 16384 x 32768 float32, which is 2 GiB on disk
- `x`: 32768 float32
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_node.input_matrix.fixed_matrix_vector_multiplication import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    TEST_COLS,
    TEST_ROWS,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.models import (
    BenchmarkSpec,
    DEFAULT_ACCUMULATION_PRECISION,
    SUPPORTED_ACCUMULATION_PRECISIONS,
)

# These scoring constants keep the score linear while still leaving room for
# slower CPUs and future accelerators to spread out meaningfully.
DEFAULT_IDEAL_SECONDS = 0.50
DEFAULT_ZERO_SCORE_SECONDS = 30.0


def build_benchmark_spec(
    *,
    rows: int | None = None,
    cols: int | None = None,
    default_variant: str = "runtime",
    ideal_seconds: float = DEFAULT_IDEAL_SECONDS,
    zero_score_seconds: float = DEFAULT_ZERO_SCORE_SECONDS,
    accumulation_precision: str = DEFAULT_ACCUMULATION_PRECISION,
) -> BenchmarkSpec:
    """Return the benchmark shape.

    The benchmark defaults to the fixed 2 GiB matrix. `rows` and `cols` stay as
    optional overrides so tests can use a tiny dataset without allocating the
    full production-sized matrix.
    """

    if rows is None:
        resolved_rows = TEST_ROWS if default_variant == "test" else DEFAULT_ROWS
    else:
        resolved_rows = rows
    if cols is None:
        resolved_cols = TEST_COLS if default_variant == "test" else DEFAULT_COLS
    else:
        resolved_cols = cols

    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    if ideal_seconds <= 0:
        raise ValueError("ideal_seconds must be positive")
    if zero_score_seconds <= ideal_seconds:
        raise ValueError("zero_score_seconds must be greater than ideal_seconds")
    if accumulation_precision not in SUPPORTED_ACCUMULATION_PRECISIONS:
        raise ValueError(
            f"accumulation_precision must be one of {', '.join(SUPPORTED_ACCUMULATION_PRECISIONS)}"
        )

    return BenchmarkSpec(
        name=f"fixed-fmvm-{resolved_rows}x{resolved_cols}",
        rows=resolved_rows,
        cols=resolved_cols,
        ideal_seconds=ideal_seconds,
        zero_score_seconds=zero_score_seconds,
        accumulation_precision=accumulation_precision,
    )
