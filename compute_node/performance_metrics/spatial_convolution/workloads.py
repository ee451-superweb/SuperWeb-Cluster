"""Describe default workload sizes for the spatial-convolution benchmark.

Use this module when the spatial benchmark needs the canonical small, medium,
large, or custom Conv2D workload specification.
"""

from __future__ import annotations

from compute_node.performance_metrics.spatial_convolution.models import BenchmarkSpec
from compute_node.input_matrix.spatial_convolution.spec import (
    MEDIUM_CIN,
    MEDIUM_COUT,
    MEDIUM_H,
    MEDIUM_K,
    MEDIUM_PAD,
    MEDIUM_STRIDE,
    MEDIUM_W,
    RUNTIME_CIN,
    RUNTIME_COUT,
    RUNTIME_H,
    RUNTIME_K,
    RUNTIME_PAD,
    RUNTIME_STRIDE,
    RUNTIME_W,
    TEST_CIN,
    TEST_COUT,
    TEST_H,
    TEST_K,
    TEST_PAD,
    TEST_STRIDE,
    TEST_W,
)

TEST_IDEAL_SECONDS = 0.5
RUNTIME_IDEAL_SECONDS = 60.0

def get_test_spec() -> BenchmarkSpec:
    """Return the small autotune-oriented spatial benchmark specification.

    Args:
        None.

    Returns:
        The canonical small Conv2D benchmark specification.
    """
    return BenchmarkSpec(
        name=f"test-conv2d-{TEST_H}x{TEST_W}",
        h=TEST_H, w=TEST_W, c_in=TEST_CIN, c_out=TEST_COUT, k=TEST_K, pad=TEST_PAD,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10, stride=TEST_STRIDE,
    )

def get_runtime_spec() -> BenchmarkSpec:
    """Return the large runtime-oriented spatial benchmark specification.

    Args:
        None.

    Returns:
        The canonical large Conv2D benchmark specification.
    """
    return BenchmarkSpec(
        name=f"runtime-conv2d-{RUNTIME_H}x{RUNTIME_W}",
        h=RUNTIME_H, w=RUNTIME_W, c_in=RUNTIME_CIN, c_out=RUNTIME_COUT, k=RUNTIME_K, pad=RUNTIME_PAD,
        ideal_seconds=RUNTIME_IDEAL_SECONDS, zero_score_seconds=RUNTIME_IDEAL_SECONDS * 10, stride=RUNTIME_STRIDE,
    )

def get_medium_spec() -> BenchmarkSpec:
    """Return the medium spatial benchmark specification.

    Args:
        None.

    Returns:
        The canonical medium Conv2D benchmark specification.
    """
    return BenchmarkSpec(
        name=f"medium-conv2d-{MEDIUM_H}x{MEDIUM_W}",
        h=MEDIUM_H, w=MEDIUM_W, c_in=MEDIUM_CIN, c_out=MEDIUM_COUT, k=MEDIUM_K, pad=MEDIUM_PAD,
        ideal_seconds=RUNTIME_IDEAL_SECONDS / 4.0, zero_score_seconds=(RUNTIME_IDEAL_SECONDS / 4.0) * 10, stride=MEDIUM_STRIDE,
    )

def build_benchmark_spec(*, h=None, w=None, c_in=None, c_out=None, k=None, pad=None, stride=None, default_variant: str = "small", **kwargs) -> BenchmarkSpec:
    """Build a custom or named spatial benchmark specification.

    Use this when the benchmark should resolve a specific CLI override or fall
    back to one of the named default workload variants.

    Args:
        h: Optional input height override.
        w: Optional input width override.
        c_in: Optional input-channel override.
        c_out: Optional output-channel override.
        k: Optional kernel-size override.
        pad: Optional padding override.
        stride: Optional stride override.
        default_variant: Named workload variant used for fallback defaults.
        **kwargs: Ignored compatibility keyword arguments.

    Returns:
        The resolved spatial benchmark specification.
    """
    if all(v is None for v in [h, w, c_in, c_out, k, pad, stride]):
        if default_variant == "medium":
            return get_medium_spec()
        if default_variant == "runtime":
            return get_runtime_spec()
        return get_test_spec()
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
    return BenchmarkSpec(
        name="custom-conv2d",
        h=h if h is not None else default_h,
        w=w if w is not None else default_w,
        c_in=c_in if c_in is not None else default_c_in,
        c_out=c_out if c_out is not None else default_c_out,
        k=k if k is not None else default_k,
        pad=pad if pad is not None else default_pad,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10,
        stride=stride if stride is not None else default_stride,
    )
