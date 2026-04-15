"""Dataset shape and metadata constants for spatial-convolution input generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT_SEED = 0x123456789ABCDEF0
DEFAULT_WEIGHT_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 8_388_608  # 32 MiB per batch
GENERATOR_ALGORITHM = "splitmix64_counter_to_float32_words_v1"

TEST_H, TEST_W = 256, 256
TEST_CIN, TEST_COUT = 32, 64
TEST_K, TEST_PAD = 3, 1
TEST_STRIDE = 1

RUNTIME_H, RUNTIME_W = 2048, 2048
RUNTIME_CIN, RUNTIME_COUT = 128, 256
RUNTIME_K, RUNTIME_PAD = 3, 1
RUNTIME_STRIDE = 1


@dataclass(slots=True)
class SpatialConvolutionSpec:
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
        return self.h * self.w * self.c_in * 4

    @property
    def weight_bytes(self) -> int:
        return self.k * self.k * self.c_in * self.c_out * 4

    @property
    def output_h(self) -> int:
        return (self.h + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_w(self) -> int:
        return (self.w + 2 * self.pad - self.k) // self.stride + 1

    @property
    def output_bytes(self) -> int:
        return self.output_h * self.output_w * self.c_out * 4


@dataclass(slots=True)
class DatasetLayout:
    root_dir: Path
    input_path: Path
    weight_path: Path
    meta_path: Path


def get_test_input_matrix_spec() -> SpatialConvolutionSpec:
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


def get_runtime_input_matrix_spec() -> SpatialConvolutionSpec:
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
    if all(value is None for value in (h, w, c_in, c_out, k, pad, stride)):
        base_spec = (
            get_runtime_input_matrix_spec()
            if default_variant == "runtime"
            else get_test_input_matrix_spec()
        )
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

    return SpatialConvolutionSpec(
        name="custom-conv2d",
        h=TEST_H if h is None else h,
        w=TEST_W if w is None else w,
        c_in=TEST_CIN if c_in is None else c_in,
        c_out=TEST_COUT if c_out is None else c_out,
        k=TEST_K if k is None else k,
        pad=TEST_PAD if pad is None else pad,
        stride=TEST_STRIDE if stride is None else stride,
    )


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    return DatasetLayout(
        root_dir=root_dir,
        input_path=root_dir / f"{prefix}input.bin",
        weight_path=root_dir / f"{prefix}weight.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )
