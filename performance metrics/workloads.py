"""Benchmark constants for the fixed matrix-vector workload.

This module is intentionally small. Its whole job is to answer:

- what shape should `A` and `x` have by default?
- what score window should we use when mapping latency to a linear score?

The default matrix shape below is exactly the one requested by the user:

- `A`: 16384 x 32768 float32, which is 2 GiB on disk
- `x`: 32768 float32
"""

from __future__ import annotations

from models import BenchmarkSpec

DEFAULT_ROWS = 16_384
DEFAULT_COLS = 32_768

# These scoring constants keep the score linear while still leaving room for
# slower CPUs and future accelerators to spread out meaningfully.
DEFAULT_IDEAL_SECONDS = 0.50
DEFAULT_ZERO_SCORE_SECONDS = 30.0


def build_benchmark_spec(
    *,
    rows: int | None = None,
    cols: int | None = None,
    ideal_seconds: float = DEFAULT_IDEAL_SECONDS,
    zero_score_seconds: float = DEFAULT_ZERO_SCORE_SECONDS,
) -> BenchmarkSpec:
    """Return the benchmark shape.

    The benchmark defaults to the fixed 2 GiB matrix. `rows` and `cols` stay as
    optional overrides so tests can use a tiny dataset without allocating the
    full production-sized matrix.
    """

    resolved_rows = DEFAULT_ROWS if rows is None else rows
    resolved_cols = DEFAULT_COLS if cols is None else cols

    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    if ideal_seconds <= 0:
        raise ValueError("ideal_seconds must be positive")
    if zero_score_seconds <= ideal_seconds:
        raise ValueError("zero_score_seconds must be greater than ideal_seconds")

    return BenchmarkSpec(
        name=f"fixed-fmvm-{resolved_rows}x{resolved_cols}",
        rows=resolved_rows,
        cols=resolved_cols,
        ideal_seconds=ideal_seconds,
        zero_score_seconds=zero_score_seconds,
    )
