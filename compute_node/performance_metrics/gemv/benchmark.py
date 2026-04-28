"""Run the top-level benchmark in GEMV-only mode.

Use this wrapper when callers want the convenience of a method-local entrypoint
without reimplementing the shared top-level benchmark CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.process import enable_utf8_mode

enable_utf8_mode()

from core.constants import METHOD_GEMV
from compute_node.performance_metrics.benchmark import main as run_top_level_benchmark
from compute_node.performance_metrics.gemv.config import RESULT_PATH


def _build_args(argv: list[str] | None = None) -> list[str]:
    """Inject GEMV-specific defaults into the forwarded CLI arguments.

    Use this before delegating to the top-level benchmark entrypoint so callers
    get the GEMV method selector and default output path automatically.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        The forwarded argument list for the top-level benchmark runner.
    """
    provided = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in provided:
        provided = ["--method", METHOD_GEMV, *provided]
    if "--output" not in provided:
        provided = [*provided, "--output", str(RESULT_PATH)]
    return provided


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the GEMV-only benchmark wrapper.

    Args:
        argv: Optional CLI argument override. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code from the delegated top-level benchmark entrypoint.
    """
    return run_top_level_benchmark(_build_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
