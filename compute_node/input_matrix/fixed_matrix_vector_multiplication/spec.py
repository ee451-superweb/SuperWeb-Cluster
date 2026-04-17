"""Dataset shape and metadata constants for FMVM input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROWS = 16_384
DEFAULT_COLS = 32_768
TEST_ROWS = 4_096
TEST_COLS = 8_192
MEDIUM_ROWS = 8_192
MEDIUM_COLS = 16_384
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
        """Return the byte size of the FMVM matrix file for this spec."""
        return self.rows * self.cols * 4

    @property
    def vector_bytes(self) -> int:
        """Return the byte size of the FMVM vector file for this spec."""
        return self.cols * 4


@dataclass(slots=True)
class DatasetLayout:
    """File locations for one generated FMVM dataset variant."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path


def get_test_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical small FMVM dataset specification."""
    return InputMatrixSpec(rows=TEST_ROWS, cols=TEST_COLS)


def get_medium_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical medium FMVM dataset specification."""
    return InputMatrixSpec(rows=MEDIUM_ROWS, cols=MEDIUM_COLS)


def get_runtime_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical large runtime FMVM dataset specification."""
    return InputMatrixSpec(rows=DEFAULT_ROWS, cols=DEFAULT_COLS)


def build_input_matrix_spec(
    *,
    rows: int | None = None,
    cols: int | None = None,
    default_variant: str = "runtime",
) -> InputMatrixSpec:
    """Return a runtime or test FMVM dataset shape."""

    if rows is None and cols is None:
        if default_variant in {"test", "small"}:
            base_spec = get_test_input_matrix_spec()
        elif default_variant == "medium":
            base_spec = get_medium_input_matrix_spec()
        else:
            base_spec = get_runtime_input_matrix_spec()
        return InputMatrixSpec(rows=base_spec.rows, cols=base_spec.cols)

    if default_variant in {"test", "small"}:
        default_rows = TEST_ROWS
        default_cols = TEST_COLS
    elif default_variant == "medium":
        default_rows = MEDIUM_ROWS
        default_cols = MEDIUM_COLS
    else:
        default_rows = DEFAULT_ROWS
        default_cols = DEFAULT_COLS
    resolved_rows = rows if rows is not None else default_rows
    resolved_cols = cols if cols is not None else default_cols
    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    return InputMatrixSpec(rows=resolved_rows, cols=resolved_cols)


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    """Build the file layout for one FMVM dataset variant.

    Args:
        root_dir: Directory that should contain the dataset files.
        prefix: Optional filename prefix such as ``test_`` or ``runtime_``.

    Returns:
        The dataset layout describing matrix, vector, and metadata paths.
    """
    return DatasetLayout(
        root_dir=root_dir,
        matrix_path=root_dir / f"{prefix}A.bin",
        vector_path=root_dir / f"{prefix}x.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )
