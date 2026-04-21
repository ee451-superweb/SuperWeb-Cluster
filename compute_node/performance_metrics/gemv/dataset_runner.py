"""Resolve and generate GEMV benchmark datasets on demand.

Use this module when the GEMV benchmark needs to choose a dataset directory or
ensure the required small, mid, and large datasets exist on disk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from adapters.process import python_utf8_command
from setup import active_python_path
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.input_matrix.gemv import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_prefix_for_size,
    dataset_is_generated,
    get_large_input_matrix_spec,
    get_mid_input_matrix_spec,
)

from compute_node.performance_metrics.path_utils import to_relative_cli_path


def resolve_dataset_dir(args, spec, *, default_dataset_dir: Path) -> Path:
    """Choose where this benchmark run should read or create ``A.bin`` and ``x.bin``.

    Use this when preparing an GEMV benchmark run so custom row/column overrides
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
    small_rows: int,
    small_cols: int,
    *,
    generate_small_dataset: bool,
    generate_mid_dataset: bool,
    generate_large_dataset: bool,
    root_dir: Path,
    generate_script_path: Path,
) -> bool:
    """Call ``generate.py`` when one or more GEMV datasets are missing.

    Use this before benchmarking so each selected workload mode has its input
    files on disk without regenerating datasets unnecessarily.

    Args:
        dataset_dir: Directory that should hold the generated datasets.
        small_rows: Row count for the small dataset variant.
        small_cols: Column count for the small dataset variant.
        generate_small_dataset: Whether the small dataset is required.
        generate_mid_dataset: Whether the mid dataset is required.
        generate_large_dataset: Whether the large dataset is required.
        root_dir: Working directory used to launch the generator script.
        generate_script_path: Path to the GEMV ``generate.py`` entrypoint.

    Returns:
        ``True`` when dataset generation actually ran, else ``False``.
    """

    small_spec = build_input_matrix_spec(rows=small_rows, cols=small_cols, default_variant="small")
    small_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("small"))
    mid_spec = get_mid_input_matrix_spec()
    mid_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("mid"))
    large_spec = get_large_input_matrix_spec()
    large_layout = build_dataset_layout(dataset_dir, prefix=dataset_prefix_for_size("large"))

    has_small = dataset_is_generated(small_layout, small_spec) if generate_small_dataset else True
    has_mid = dataset_is_generated(mid_layout, mid_spec) if generate_mid_dataset else True
    has_large = dataset_is_generated(large_layout, large_spec) if generate_large_dataset else True
    if has_small and has_mid and has_large:
        return False

    command = python_utf8_command(
        active_python_path(),
        to_relative_cli_path(generate_script_path, start=root_dir),
        "--output-dir",
        to_relative_cli_path(dataset_dir, start=root_dir),
        "--rows",
        str(small_rows),
        "--cols",
        str(small_cols),
    )
    if not generate_small_dataset:
        command.append("--skip-small")
    if not generate_mid_dataset:
        command.append("--skip-mid")
    if not generate_large_dataset:
        command.append("--skip-large")
    emit_status(
        "method.gemv.dataset.generate.start",
        status="running",
        method="gemv",
        dataset_dir=str(dataset_dir),
        small_rows=small_rows,
        small_cols=small_cols,
        generate_small_dataset=generate_small_dataset,
        generate_mid_dataset=generate_mid_dataset,
        generate_large_dataset=generate_large_dataset,
        command=command,
        cwd=str(root_dir),
    )
    subprocess.run(command, check=True, cwd=root_dir)
    emit_status(
        "method.gemv.dataset.generate.complete",
        status="running",
        method="gemv",
        dataset_dir=str(dataset_dir),
        small_rows=small_rows,
        small_cols=small_cols,
        generate_small_dataset=generate_small_dataset,
        generate_mid_dataset=generate_mid_dataset,
        generate_large_dataset=generate_large_dataset,
    )

    generated_small = dataset_is_generated(small_layout, small_spec) if generate_small_dataset else True
    generated_mid = dataset_is_generated(mid_layout, mid_spec) if generate_mid_dataset else True
    generated_large = dataset_is_generated(large_layout, large_spec) if generate_large_dataset else True
    if not generated_small or not generated_mid or not generated_large:
        raise RuntimeError("generate.py completed but the generated dataset is still incomplete")
    return True
