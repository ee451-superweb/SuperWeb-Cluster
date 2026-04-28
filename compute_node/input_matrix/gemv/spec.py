"""Dataset shape and metadata constants for GEMV input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LARGE_ROWS = 16_384
LARGE_COLS = 32_768
SMALL_ROWS = 2_048
SMALL_COLS = 4_096
MID_ROWS = 8_192
MID_COLS = 16_384
DEFAULT_MATRIX_SEED = 0x123456789ABCDEF0
DEFAULT_VECTOR_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
PROGRESS_STEP_BYTES = 32 * 1024 * 1024
HASH_READ_BYTES = 16 * 1024 * 1024
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"
WORKLOAD_SIZE_SMALL = "small"
WORKLOAD_SIZE_MID = "mid"
WORKLOAD_SIZE_LARGE = "large"
LEGACY_SIZE_ALIASES = {
    "test": WORKLOAD_SIZE_SMALL,
    "medium": WORKLOAD_SIZE_MID,
    "runtime": WORKLOAD_SIZE_LARGE,
}
DEFAULT_ROWS = LARGE_ROWS
DEFAULT_COLS = LARGE_COLS
TEST_ROWS = SMALL_ROWS
TEST_COLS = SMALL_COLS
MEDIUM_ROWS = MID_ROWS
MEDIUM_COLS = MID_COLS


@dataclass(slots=True)
class InputMatrixSpec:
    """Shape information for one generated matrix/vector dataset."""

    rows: int
    cols: int

    @property
    def matrix_bytes(self) -> int:
        """Return the byte size of the GEMV matrix file for this spec."""
        return self.rows * self.cols * 4

    @property
    def vector_bytes(self) -> int:
        """Return the byte size of the GEMV vector file for this spec."""
        return self.cols * 4


@dataclass(slots=True)
class DatasetLayout:
    """File locations for one generated GEMV dataset variant."""

    root_dir: Path
    matrix_path: Path
    vector_path: Path
    meta_path: Path


def normalize_size_variant(size: str | None, *, default: str = WORKLOAD_SIZE_LARGE) -> str:
    """Normalize one requested GEMV size string into the canonical variant name."""

    normalized = (size or "").strip().lower()
    if not normalized:
        return default
    return LEGACY_SIZE_ALIASES.get(normalized, normalized)


def dataset_prefix_for_size(size: str | None, *, default: str = WORKLOAD_SIZE_LARGE) -> str:
    """Return the filename prefix used for one named GEMV dataset size."""

    return f"{normalize_size_variant(size, default=default)}_"


def get_small_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical small GEMV dataset specification."""
    return InputMatrixSpec(rows=SMALL_ROWS, cols=SMALL_COLS)


def get_mid_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical mid-sized GEMV dataset specification."""
    return InputMatrixSpec(rows=MID_ROWS, cols=MID_COLS)


def get_large_input_matrix_spec() -> InputMatrixSpec:
    """Return the canonical large GEMV dataset specification."""
    return InputMatrixSpec(rows=LARGE_ROWS, cols=LARGE_COLS)


def get_test_input_matrix_spec() -> InputMatrixSpec:
    """Return the legacy small GEMV dataset specification wrapper."""
    return get_small_input_matrix_spec()


def get_medium_input_matrix_spec() -> InputMatrixSpec:
    """Return the legacy mid-sized GEMV dataset specification wrapper."""
    return get_mid_input_matrix_spec()


def get_runtime_input_matrix_spec() -> InputMatrixSpec:
    """Return the legacy large GEMV dataset specification wrapper."""
    return get_large_input_matrix_spec()


def build_input_matrix_spec(
    *,
    rows: int | None = None,
    cols: int | None = None,
    default_variant: str = WORKLOAD_SIZE_LARGE,
) -> InputMatrixSpec:
    """Return a runtime or test GEMV dataset shape."""

    default_variant = normalize_size_variant(default_variant)
    if rows is None and cols is None:
        if default_variant == WORKLOAD_SIZE_SMALL:
            base_spec = get_small_input_matrix_spec()
        elif default_variant == WORKLOAD_SIZE_MID:
            base_spec = get_mid_input_matrix_spec()
        else:
            base_spec = get_large_input_matrix_spec()
        return InputMatrixSpec(rows=base_spec.rows, cols=base_spec.cols)

    if default_variant == WORKLOAD_SIZE_SMALL:
        default_rows = SMALL_ROWS
        default_cols = SMALL_COLS
    elif default_variant == WORKLOAD_SIZE_MID:
        default_rows = MID_ROWS
        default_cols = MID_COLS
    else:
        default_rows = LARGE_ROWS
        default_cols = LARGE_COLS
    resolved_rows = rows if rows is not None else default_rows
    resolved_cols = cols if cols is not None else default_cols
    if resolved_rows <= 0 or resolved_cols <= 0:
        raise ValueError("rows and cols must be positive")
    return InputMatrixSpec(rows=resolved_rows, cols=resolved_cols)


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    """Build the file layout for one GEMV dataset variant.

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
