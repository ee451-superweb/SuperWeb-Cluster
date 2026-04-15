"""Compatibility wrapper for the method-local spatial-convolution dataset CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compute_node.input_matrix.spatial_convolution.generate import main


if __name__ == "__main__":
    raise SystemExit(main())
