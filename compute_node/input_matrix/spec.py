"""Shared dataset shape and metadata constants for `compute_node/input_matrix/`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROWS = 16_384
DEFAULT_COLS = 32_768
DEFAULT_MATRIX_SEED = 0x123456789ABCDEF0
DEFAULT_VECTOR_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
PROGRESS_STEP_BYTES = 32 * 1024 * 1024
HASH_READ_BYTES = 16 * 1024 * 1024
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"


@dataclass(slots=True)
class InputMatrixSpec:
    """Shape information for the generated matrix/vector dataset."""

    rows: int
    cols: int

    @property
    def matrix_bytes(self) -> int:
        """Return the on-disk size of `A.bin`."""

        return self.rows * self.cols * 4

    @property
    def vector_bytes(self) -> int:
        """Return the on-disk size of `x.bin`."""

        return self.cols * 4


@dataclass(slots=True)
class DatasetLayout:
    """File locations for the generated input dataset."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path


def build_input_matrix_spec(*, rows: int | None = None, cols: int | None = None) -> InputMatrixSpec:
    """Return the default or overridden input-matrix shape."""

    resolved_rows = DEFAULT_ROWS if rows is None else rows
    resolved_cols = DEFAULT_COLS if cols is None else cols
    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    return InputMatrixSpec(rows=resolved_rows, cols=resolved_cols)


def build_dataset_layout(root_dir: Path) -> DatasetLayout:
    """Return the canonical paths for the generated dataset files."""

    return DatasetLayout(
        root_dir=root_dir,
        matrix_path=root_dir / "A.bin",
        vector_path=root_dir / "x.bin",
        meta_path=root_dir / "dataset_meta.json",
    )
