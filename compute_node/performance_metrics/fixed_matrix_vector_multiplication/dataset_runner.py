"""Resolve and generate FMVM benchmark datasets on demand.

Use this module when the FMVM benchmark needs to choose a dataset directory or
ensure the required small, medium, and large datasets exist on disk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.input_matrix.fixed_matrix_vector_multiplication import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    get_medium_input_matrix_spec,
    get_runtime_input_matrix_spec,
)

from compute_node.performance_metrics.path_utils import to_relative_cli_path


def resolve_dataset_dir(args, spec, *, default_dataset_dir: Path) -> Path:
    """Choose where this benchmark run should read or create ``A.bin`` and ``x.bin``.

    Use this when preparing an FMVM benchmark run so custom row/column overrides
    can live under an isolated subdirectory without disturbing the default data.

    Args:
        args: Parsed benchmark CLI arguments.
        spec: Resolved benchmark specification for this run.
        default_dataset_dir: Canonical dataset directory for non-override runs.

    Returns:
        The dataset directory that this benchmark run should use.
    """

    requested_dir = Path(args.dataset_dir)
    if (args.rows is None and args.cols is None) or requested_dir != default_dataset_dir:
        return requested_dir

    return requested_dir / "overrides" / f"{spec.rows}x{spec.cols}"


def generate_dataset_if_missing(
    dataset_dir: Path,
    test_rows: int,
    test_cols: int,
    *,
    generate_small_dataset: bool,
    generate_medium_dataset: bool,
    generate_large_dataset: bool,
    root_dir: Path,
    generate_script_path: Path,
) -> bool:
    """Call ``generate.py`` when one or more FMVM datasets are missing.

    Use this before benchmarking so each selected workload mode has its input
    files on disk without regenerating datasets unnecessarily.

    Args:
        dataset_dir: Directory that should hold the generated datasets.
        test_rows: Row count for the small dataset variant.
        test_cols: Column count for the small dataset variant.
        generate_small_dataset: Whether the small dataset is required.
        generate_medium_dataset: Whether the medium dataset is required.
        generate_large_dataset: Whether the large dataset is required.
        root_dir: Working directory used to launch the generator script.
        generate_script_path: Path to the FMVM ``generate.py`` entrypoint.

    Returns:
        ``True`` when dataset generation actually ran, else ``False``.
    """

    test_spec = build_input_matrix_spec(rows=test_rows, cols=test_cols, default_variant="test")
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    medium_spec = get_medium_input_matrix_spec()
    medium_layout = build_dataset_layout(dataset_dir, prefix="medium_")
    runtime_spec = get_runtime_input_matrix_spec()
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")

    has_test = dataset_is_generated(test_layout, test_spec) if generate_small_dataset else True
    has_medium = dataset_is_generated(medium_layout, medium_spec) if generate_medium_dataset else True
    has_runtime = dataset_is_generated(runtime_layout, runtime_spec) if generate_large_dataset else True
    if has_test and has_medium and has_runtime:
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
    if not generate_small_dataset:
        command.append("--skip-small")
    if not generate_medium_dataset:
        command.append("--skip-medium")
    if not generate_large_dataset:
        command.append("--skip-large")
    emit_status(
        "method.fmvm.dataset.generate.start",
        status="running",
        method="fixed_matrix_vector_multiplication",
        dataset_dir=str(dataset_dir),
        test_rows=test_rows,
        test_cols=test_cols,
        generate_small_dataset=generate_small_dataset,
        generate_medium_dataset=generate_medium_dataset,
        generate_large_dataset=generate_large_dataset,
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
        generate_small_dataset=generate_small_dataset,
        generate_medium_dataset=generate_medium_dataset,
        generate_large_dataset=generate_large_dataset,
    )

    generated_test = dataset_is_generated(test_layout, test_spec) if generate_small_dataset else True
    generated_medium = dataset_is_generated(medium_layout, medium_spec) if generate_medium_dataset else True
    generated_runtime = dataset_is_generated(runtime_layout, runtime_spec) if generate_large_dataset else True
    if not generated_test or not generated_medium or not generated_runtime:
        raise RuntimeError("generate.py completed but the generated dataset is still incomplete")
    return True
