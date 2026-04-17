"""Dataset shape and metadata constants for spatial-convolution input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT_SEED = 0x123456789ABCDEF0
DEFAULT_WEIGHT_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"

TEST_H, TEST_W = 512, 512
TEST_CIN, TEST_COUT = 64, 128
TEST_K, TEST_PAD = 3, 1
TEST_STRIDE = 1

MEDIUM_H, MEDIUM_W = 1024, 1024
MEDIUM_CIN, MEDIUM_COUT = 128, 256
MEDIUM_K, MEDIUM_PAD = 3, 1
MEDIUM_STRIDE = 1

RUNTIME_H, RUNTIME_W = 2048, 2048
RUNTIME_CIN, RUNTIME_COUT = 128, 256
RUNTIME_K, RUNTIME_PAD = 3, 1
RUNTIME_STRIDE = 1


@dataclass(slots=True)
class SpatialConvolutionSpec:
    """Describe one spatial-convolution dataset shape."""

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
    """Describe the file layout for one spatial dataset variant."""

    root_dir: Path
    input_path: Path
    weight_path: Path
    meta_path: Path


def get_test_input_matrix_spec() -> SpatialConvolutionSpec:
    """Return the canonical small spatial dataset specification."""
    return SpatialConvolutionSpec(
        name=f"test-conv2d-{TEST_H}x{TEST_W}",
        h=TEST_H,
        w=TEST_W,
        c_in=TEST_CIN,
        c_out=TEST_COUT,
        k=TEST_K,
        pad=TEST_PAD,
        stride=TEST_STRIDE,
    )


def get_medium_input_matrix_spec() -> SpatialConvolutionSpec:
    """Return the canonical medium spatial dataset specification."""
    return SpatialConvolutionSpec(
        name=f"medium-conv2d-{MEDIUM_H}x{MEDIUM_W}",
        h=MEDIUM_H,
        w=MEDIUM_W,
        c_in=MEDIUM_CIN,
        c_out=MEDIUM_COUT,
        k=MEDIUM_K,
        pad=MEDIUM_PAD,
        stride=MEDIUM_STRIDE,
    )


def get_runtime_input_matrix_spec() -> SpatialConvolutionSpec:
    """Return the canonical large runtime spatial dataset specification."""
    return SpatialConvolutionSpec(
        name=f"runtime-conv2d-{RUNTIME_H}x{RUNTIME_W}",
        h=RUNTIME_H,
        w=RUNTIME_W,
        c_in=RUNTIME_CIN,
        c_out=RUNTIME_COUT,
        k=RUNTIME_K,
        pad=RUNTIME_PAD,
        stride=RUNTIME_STRIDE,
    )


def build_input_matrix_spec(
    *,
    h: int | None = None,
    w: int | None = None,
    c_in: int | None = None,
    c_out: int | None = None,
    k: int | None = None,
    pad: int | None = None,
    stride: int | None = None,
    default_variant: str = "test",
) -> SpatialConvolutionSpec:
    """Build a custom or named spatial dataset specification.

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
        The resolved spatial dataset specification.
    """
    if all(value is None for value in (h, w, c_in, c_out, k, pad, stride)):
        if default_variant in {"test", "small"}:
            base_spec = get_test_input_matrix_spec()
        elif default_variant == "medium":
            base_spec = get_medium_input_matrix_spec()
        else:
            base_spec = get_runtime_input_matrix_spec()
        return SpatialConvolutionSpec(
            name=base_spec.name,
            h=base_spec.h,
            w=base_spec.w,
            c_in=base_spec.c_in,
            c_out=base_spec.c_out,
            k=base_spec.k,
            pad=base_spec.pad,
            stride=base_spec.stride,
        )

    if default_variant == "medium":
        default_h, default_w = MEDIUM_H, MEDIUM_W
        default_c_in, default_c_out = MEDIUM_CIN, MEDIUM_COUT
        default_k, default_pad, default_stride = MEDIUM_K, MEDIUM_PAD, MEDIUM_STRIDE
    elif default_variant == "runtime":
        default_h, default_w = RUNTIME_H, RUNTIME_W
        default_c_in, default_c_out = RUNTIME_CIN, RUNTIME_COUT
        default_k, default_pad, default_stride = RUNTIME_K, RUNTIME_PAD, RUNTIME_STRIDE
    else:
        default_h, default_w = TEST_H, TEST_W
        default_c_in, default_c_out = TEST_CIN, TEST_COUT
        default_k, default_pad, default_stride = TEST_K, TEST_PAD, TEST_STRIDE
    return SpatialConvolutionSpec(
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
    """Build the file layout for one spatial dataset variant.

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
