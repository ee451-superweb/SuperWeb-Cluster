"""Dataset shape and metadata constants for conv2d input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT_SEED = 0x123456789ABCDEF0
DEFAULT_WEIGHT_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"

SMALL_H, SMALL_W = 256, 256
SMALL_CIN, SMALL_COUT = 32, 64
SMALL_K, SMALL_PAD = 3, 1
SMALL_STRIDE = 1

MID_H, MID_W = 1024, 1024
MID_CIN, MID_COUT = 128, 256
MID_K, MID_PAD = 3, 1
MID_STRIDE = 1

LARGE_H, LARGE_W = 2048, 2048
LARGE_CIN, LARGE_COUT = 128, 256
LARGE_K, LARGE_PAD = 3, 1
LARGE_STRIDE = 1
WORKLOAD_SIZE_SMALL = "small"
WORKLOAD_SIZE_MID = "mid"
WORKLOAD_SIZE_LARGE = "large"
LEGACY_SIZE_ALIASES = {
    "test": WORKLOAD_SIZE_SMALL,
    "medium": WORKLOAD_SIZE_MID,
    "runtime": WORKLOAD_SIZE_LARGE,
}
TEST_H, TEST_W = SMALL_H, SMALL_W
TEST_CIN, TEST_COUT = SMALL_CIN, SMALL_COUT
TEST_K, TEST_PAD = SMALL_K, SMALL_PAD
TEST_STRIDE = SMALL_STRIDE
MEDIUM_H, MEDIUM_W = MID_H, MID_W
MEDIUM_CIN, MEDIUM_COUT = MID_CIN, MID_COUT
MEDIUM_K, MEDIUM_PAD = MID_K, MID_PAD
MEDIUM_STRIDE = MID_STRIDE
RUNTIME_H, RUNTIME_W = LARGE_H, LARGE_W
RUNTIME_CIN, RUNTIME_COUT = LARGE_CIN, LARGE_COUT
RUNTIME_K, RUNTIME_PAD = LARGE_K, LARGE_PAD
RUNTIME_STRIDE = LARGE_STRIDE


@dataclass(slots=True)
class Conv2dSpec:
    """Describe one conv2d dataset shape."""

    name: str
    h: int
    w: int
    c_in: int
    c_out: int
    k: int
    pad: int
    stride: int = 1

    @property
    def input_bytes(self) -> int:
        """Return the byte size of the input tensor for this spec."""
        return self.h * self.w * self.c_in * 4

    @property
    def weight_bytes(self) -> int:
        """Return the byte size of the weight tensor for this spec."""
        return self.k * self.k * self.c_in * self.c_out * 4

    @property
    def output_h(self) -> int:
        """Return the output height implied by the convolution shape."""
        return (self.h + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_w(self) -> int:
        """Return the output width implied by the convolution shape."""
        return (self.w + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_bytes(self) -> int:
        """Return the byte size of the output tensor for this spec."""
        return self.output_h * self.output_w * self.c_out * 4


@dataclass(slots=True)
class DatasetLayout:
    """Describe the file layout for one conv2d dataset variant."""

    root_dir: Path
    input_path: Path
    weight_path: Path
    meta_path: Path


def normalize_size_variant(size: str | None, *, default: str = WORKLOAD_SIZE_SMALL) -> str:
    """Normalize one requested conv2d size string into the canonical variant name."""

    normalized = (size or "").strip().lower()
    if not normalized:
        return default
    return LEGACY_SIZE_ALIASES.get(normalized, normalized)


def dataset_prefix_for_size(size: str | None, *, default: str = WORKLOAD_SIZE_SMALL) -> str:
    """Return the filename prefix used for one named conv2d dataset size."""

    return f"{normalize_size_variant(size, default=default)}_"


def get_small_input_matrix_spec() -> Conv2dSpec:
    """Return the canonical small conv2d dataset specification."""
    return Conv2dSpec(
        name=f"small-conv2d-{SMALL_H}x{SMALL_W}",
        h=SMALL_H,
        w=SMALL_W,
        c_in=SMALL_CIN,
        c_out=SMALL_COUT,
        k=SMALL_K,
        pad=SMALL_PAD,
        stride=SMALL_STRIDE,
    )


def get_mid_input_matrix_spec() -> Conv2dSpec:
    """Return the canonical mid-sized conv2d dataset specification."""
    return Conv2dSpec(
        name=f"mid-conv2d-{MID_H}x{MID_W}",
        h=MID_H,
        w=MID_W,
        c_in=MID_CIN,
        c_out=MID_COUT,
        k=MID_K,
        pad=MID_PAD,
        stride=MID_STRIDE,
    )


def get_large_input_matrix_spec() -> Conv2dSpec:
    """Return the canonical large conv2d dataset specification."""
    return Conv2dSpec(
        name=f"large-conv2d-{LARGE_H}x{LARGE_W}",
        h=LARGE_H,
        w=LARGE_W,
        c_in=LARGE_CIN,
        c_out=LARGE_COUT,
        k=LARGE_K,
        pad=LARGE_PAD,
        stride=LARGE_STRIDE,
    )


def get_test_input_matrix_spec() -> Conv2dSpec:
    """Return the legacy small conv2d dataset specification wrapper."""
    return get_small_input_matrix_spec()


def get_medium_input_matrix_spec() -> Conv2dSpec:
    """Return the legacy mid-sized conv2d dataset specification wrapper."""
    return get_mid_input_matrix_spec()


def get_runtime_input_matrix_spec() -> Conv2dSpec:
    """Return the legacy large conv2d dataset specification wrapper."""
    return get_large_input_matrix_spec()


def build_input_matrix_spec(
    *,
    h: int | None = None,
    w: int | None = None,
    c_in: int | None = None,
    c_out: int | None = None,
    k: int | None = None,
    pad: int | None = None,
    stride: int | None = None,
    default_variant: str = WORKLOAD_SIZE_SMALL,
) -> Conv2dSpec:
    """Build a custom or named conv2d dataset specification.

    Args:
        h: Optional input height override.
        w: Optional input width override.
        c_in: Optional input-channel override.
        c_out: Optional output-channel override.
        k: Optional kernel-size override.
        pad: Optional padding override.
        stride: Optional stride override.
        default_variant: Named default variant used when overrides are omitted.

    Returns:
        The resolved conv2d dataset specification.
    """
    default_variant = normalize_size_variant(default_variant, default=WORKLOAD_SIZE_SMALL)
    if all(value is None for value in (h, w, c_in, c_out, k, pad, stride)):
        if default_variant == WORKLOAD_SIZE_SMALL:
            base_spec = get_small_input_matrix_spec()
        elif default_variant == WORKLOAD_SIZE_MID:
            base_spec = get_mid_input_matrix_spec()
        else:
            base_spec = get_large_input_matrix_spec()
        return Conv2dSpec(
            name=base_spec.name,
            h=base_spec.h,
            w=base_spec.w,
            c_in=base_spec.c_in,
            c_out=base_spec.c_out,
            k=base_spec.k,
            pad=base_spec.pad,
            stride=base_spec.stride,
        )

    if default_variant == WORKLOAD_SIZE_MID:
        default_h, default_w = MID_H, MID_W
        default_c_in, default_c_out = MID_CIN, MID_COUT
        default_k, default_pad, default_stride = MID_K, MID_PAD, MID_STRIDE
    elif default_variant == WORKLOAD_SIZE_LARGE:
        default_h, default_w = LARGE_H, LARGE_W
        default_c_in, default_c_out = LARGE_CIN, LARGE_COUT
        default_k, default_pad, default_stride = LARGE_K, LARGE_PAD, LARGE_STRIDE
    else:
        default_h, default_w = SMALL_H, SMALL_W
        default_c_in, default_c_out = SMALL_CIN, SMALL_COUT
        default_k, default_pad, default_stride = SMALL_K, SMALL_PAD, SMALL_STRIDE
    return Conv2dSpec(
        name="custom-conv2d",
        h=default_h if h is None else h,
        w=default_w if w is None else w,
        c_in=default_c_in if c_in is None else c_in,
        c_out=default_c_out if c_out is None else c_out,
        k=default_k if k is None else k,
        pad=default_pad if pad is None else pad,
        stride=default_stride if stride is None else stride,
    )


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    """Build the file layout for one conv2d dataset variant.

    Args:
        root_dir: Directory that should contain the dataset files.
        prefix: Optional filename prefix such as ``test_`` or ``runtime_``.

    Returns:
        The dataset layout describing input, weight, and metadata paths.
    """
    return DatasetLayout(
        root_dir=root_dir,
        input_path=root_dir / f"{prefix}input.bin",
        weight_path=root_dir / f"{prefix}weight.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )
