"""Dataset shape and metadata constants for GEMM input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Square GEMMs keep the partition math simple; the runner slices the M axis,
# so large M gives meaningful per-worker work even with several workers.
SMALL_M = 1_024
SMALL_N = 1_024
SMALL_K = 1_024
MID_M = 4_096
MID_N = 4_096
MID_K = 4_096
LARGE_M = 8_192
LARGE_N = 8_192
LARGE_K = 8_192

DEFAULT_A_SEED = 0x1122334455667788
DEFAULT_B_SEED = 0x99AABBCCDDEEFF00
DEFAULT_CHUNK_VALUES = 8_388_608
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"

WORKLOAD_SIZE_SMALL = "small"
WORKLOAD_SIZE_MID = "mid"
WORKLOAD_SIZE_LARGE = "large"
LEGACY_SIZE_ALIASES = {
    "test": WORKLOAD_SIZE_SMALL,
    "medium": WORKLOAD_SIZE_MID,
    "runtime": WORKLOAD_SIZE_LARGE,
}


@dataclass(slots=True)
class GemmSpec:
    """Shape information for one GEMM dataset variant.

    Inputs ``A`` is ``[M, K]`` row-major float32, ``B`` is ``[K, N]``
    row-major float32. Output ``C = A @ B`` is ``[M, N]`` row-major float32.
    """

    m: int
    n: int
    k: int

    @property
    def a_bytes(self) -> int:
        """Return the byte size of the generated A input file."""
        return self.m * self.k * 4

    @property
    def b_bytes(self) -> int:
        """Return the byte size of the generated B input file."""
        return self.k * self.n * 4

    @property
    def output_bytes(self) -> int:
        """Return the byte size of the full output C (row-major [M, N] float32)."""
        return self.m * self.n * 4


@dataclass(slots=True)
class DatasetLayout:
    """File locations for one generated GEMM dataset variant."""

    root_dir: Path
    a_path: Path
    b_path: Path
    meta_path: Path


def normalize_size_variant(size: str | None, *, default: str = WORKLOAD_SIZE_LARGE) -> str:
    """Normalize one requested GEMM size string into the canonical variant name."""

    normalized = (size or "").strip().lower()
    if not normalized:
        return default
    return LEGACY_SIZE_ALIASES.get(normalized, normalized)


def dataset_prefix_for_size(size: str | None, *, default: str = WORKLOAD_SIZE_LARGE) -> str:
    """Return the filename prefix used for one named GEMM dataset size."""

    return f"{normalize_size_variant(size, default=default)}_"


def get_small_spec() -> GemmSpec:
    """Return the canonical small GEMM dataset specification."""
    return GemmSpec(m=SMALL_M, n=SMALL_N, k=SMALL_K)


def get_mid_spec() -> GemmSpec:
    """Return the canonical mid GEMM dataset specification."""
    return GemmSpec(m=MID_M, n=MID_N, k=MID_K)


def get_large_spec() -> GemmSpec:
    """Return the canonical large GEMM dataset specification."""
    return GemmSpec(m=LARGE_M, n=LARGE_N, k=LARGE_K)


def build_spec(
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    default_variant: str = WORKLOAD_SIZE_LARGE,
) -> GemmSpec:
    """Return a GEMM dataset specification for one named size or explicit dims."""

    default_variant = normalize_size_variant(default_variant)
    if m is None and n is None and k is None:
        if default_variant == WORKLOAD_SIZE_SMALL:
            return get_small_spec()
        if default_variant == WORKLOAD_SIZE_MID:
            return get_mid_spec()
        return get_large_spec()

    if default_variant == WORKLOAD_SIZE_SMALL:
        default_m, default_n, default_k = SMALL_M, SMALL_N, SMALL_K
    elif default_variant == WORKLOAD_SIZE_MID:
        default_m, default_n, default_k = MID_M, MID_N, MID_K
    else:
        default_m, default_n, default_k = LARGE_M, LARGE_N, LARGE_K
    resolved_m = m if m is not None else default_m
    resolved_n = n if n is not None else default_n
    resolved_k = k if k is not None else default_k
    if resolved_m <= 0 or resolved_n <= 0 or resolved_k <= 0:
        raise ValueError("M, N, K must be positive")
    return GemmSpec(m=resolved_m, n=resolved_n, k=resolved_k)


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    """Build the file layout for one GEMM dataset variant."""

    return DatasetLayout(
        root_dir=root_dir,
        a_path=root_dir / f"{prefix}A.bin",
        b_path=root_dir / f"{prefix}B.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )
