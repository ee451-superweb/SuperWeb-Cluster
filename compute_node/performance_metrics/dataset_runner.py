"""Dataset resolution helpers for the benchmark entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from compute_node.input_matrix import build_dataset_layout, build_input_matrix_spec, dataset_is_generated

from path_utils import to_relative_cli_path


def resolve_dataset_dir(args, spec, *, default_dataset_dir: Path) -> Path:
    """Choose where this benchmark run should read or create `A.bin` and `x.bin`."""

    requested_dir = Path(args.dataset_dir)
    if (args.rows is None and args.cols is None) or requested_dir != default_dataset_dir:
        return requested_dir

    return requested_dir / "overrides" / f"{spec.rows}x{spec.cols}"


def generate_dataset_if_missing(
    dataset_dir: Path,
    rows: int,
    cols: int,
    *,
    root_dir: Path,
    generate_script_path: Path,
) -> bool:
    """Call `generate.py` when the generated input files do not exist yet."""

    spec = build_input_matrix_spec(rows=rows, cols=cols)
    layout = build_dataset_layout(dataset_dir)
    if dataset_is_generated(layout, spec):
        return False

    command = [
        sys.executable,
        to_relative_cli_path(generate_script_path, start=root_dir),
        "--output-dir",
        to_relative_cli_path(dataset_dir, start=root_dir),
        "--rows",
        str(rows),
        "--cols",
        str(cols),
    ]
    subprocess.run(command, check=True, cwd=root_dir)

    if not dataset_is_generated(layout, spec):
        raise RuntimeError("generate.py completed but the generated dataset is still incomplete")
    return True
