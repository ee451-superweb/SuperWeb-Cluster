"""Dataset resolution helpers for the benchmark entrypoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.input_matrix.fixed_matrix_vector_multiplication import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    get_runtime_input_matrix_spec,
)

from compute_node.performance_metrics.path_utils import to_relative_cli_path


def resolve_dataset_dir(args, spec, *, default_dataset_dir: Path) -> Path:
    """Choose where this benchmark run should read or create `A.bin` and `x.bin`."""

    requested_dir = Path(args.dataset_dir)
    if (args.rows is None and args.cols is None) or requested_dir != default_dataset_dir:
        return requested_dir

    return requested_dir / "overrides" / f"{spec.rows}x{spec.cols}"


def generate_dataset_if_missing(
    dataset_dir: Path,
    test_rows: int,
    test_cols: int,
    *,
    require_runtime_measurement: bool,
    root_dir: Path,
    generate_script_path: Path,
) -> bool:
    """Call `generate.py` when one or more FMVM datasets are missing."""

    test_spec = build_input_matrix_spec(rows=test_rows, cols=test_cols, default_variant="test")
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    runtime_spec = get_runtime_input_matrix_spec()
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")

    has_test = dataset_is_generated(test_layout, test_spec)
    has_runtime = dataset_is_generated(runtime_layout, runtime_spec)
    if has_test and (not require_runtime_measurement or has_runtime):
        return False

    command = [
        str(active_python_path()),
        to_relative_cli_path(generate_script_path, start=root_dir),
        "--output-dir",
        to_relative_cli_path(dataset_dir, start=root_dir),
        "--rows",
        str(test_rows),
        "--cols",
        str(test_cols),
    ]
    if not require_runtime_measurement:
        command.append("--skip-runtime")
    emit_status(
        "method.fmvm.dataset.generate.start",
        status="running",
        method="fixed_matrix_vector_multiplication",
        dataset_dir=str(dataset_dir),
        test_rows=test_rows,
        test_cols=test_cols,
        require_runtime_measurement=require_runtime_measurement,
        command=command,
        cwd=str(root_dir),
    )
    subprocess.run(command, check=True, cwd=root_dir)
    emit_status(
        "method.fmvm.dataset.generate.complete",
        status="running",
        method="fixed_matrix_vector_multiplication",
        dataset_dir=str(dataset_dir),
        test_rows=test_rows,
        test_cols=test_cols,
        require_runtime_measurement=require_runtime_measurement,
    )

    generated_test = dataset_is_generated(test_layout, test_spec)
    generated_runtime = dataset_is_generated(runtime_layout, runtime_spec)
    if not generated_test or (require_runtime_measurement and not generated_runtime):
        raise RuntimeError("generate.py completed but the generated dataset is still incomplete")
    return True
