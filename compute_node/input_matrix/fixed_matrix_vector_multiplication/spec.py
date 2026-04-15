"""Dataset shape and metadata constants for FMVM input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROWS = 16_384
DEFAULT_COLS = 32_768
TEST_ROWS = 2_048
TEST_COLS = 4_096
DEFAULT_MATRIX_SEED = 0x123456789ABCDEF0
DEFAULT_VECTOR_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
PROGRESS_STEP_BYTES = 32 * 1024 * 1024
HASH_READ_BYTES = 16 * 1024 * 1024
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"


@dataclass(slots=True)
class InputMatrixSpec:
    """Shape information for one generated matrix/vector dataset."""

    rows: int
    cols: int

    @property
    def matrix_bytes(self) -> int:
        return self.rows * self.cols * 4

    @property
    def vector_bytes(self) -> int:
        return self.cols * 4


@dataclass(slots=True)
class DatasetLayout:
    """File locations for one generated FMVM dataset variant."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path


def get_test_input_matrix_spec() -> InputMatrixSpec:
    return InputMatrixSpec(rows=TEST_ROWS, cols=TEST_COLS)


def get_runtime_input_matrix_spec() -> InputMatrixSpec:
    return InputMatrixSpec(rows=DEFAULT_ROWS, cols=DEFAULT_COLS)


def build_input_matrix_spec(
    *,
    rows: int | None = None,
    cols: int | None = None,
    default_variant: str = "runtime",
) -> InputMatrixSpec:
    """Return a runtime or test FMVM dataset shape."""

    if rows is None and cols is None:
        base_spec = (
            get_test_input_matrix_spec()
            if default_variant == "test"
            else get_runtime_input_matrix_spec()
        )
        return InputMatrixSpec(rows=base_spec.rows, cols=base_spec.cols)

    resolved_rows = rows if rows is not None else (
        TEST_ROWS if default_variant == "test" else DEFAULT_ROWS
    )
    resolved_cols = cols if cols is not None else (
        TEST_COLS if default_variant == "test" else DEFAULT_COLS
    )
    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    return InputMatrixSpec(rows=resolved_rows, cols=resolved_cols)


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    return DatasetLayout(
        root_dir=root_dir,
        matrix_path=root_dir / f"{prefix}A.bin",
        vector_path=root_dir / f"{prefix}x.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )
