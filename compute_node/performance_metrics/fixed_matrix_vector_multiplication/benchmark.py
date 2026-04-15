"""Wrapper that runs the top-level benchmark in FMVM-only mode."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
from compute_node.performance_metrics.benchmark import main as run_top_level_benchmark
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import RESULT_PATH


def _build_args(argv: list[str] | None = None) -> list[str]:
    provided = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in provided:
        provided = ["--method", METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, *provided]
    if "--output" not in provided:
        provided = [*provided, "--output", str(RESULT_PATH)]
    return provided


def main(argv: list[str] | None = None) -> int:
    return run_top_level_benchmark(_build_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
