"""Describe default workload sizes for the conv2d benchmark.

Use this module when the conv2d benchmark needs the canonical small, mid,
large, or custom Conv2D workload specification.
"""

from __future__ import annotations

from compute_node.performance_metrics.conv2d.models import BenchmarkSpec
from compute_node.input_matrix.conv2d.spec import (
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

SMALL_IDEAL_SECONDS = 0.5
MID_IDEAL_SECONDS = 15.0
LARGE_IDEAL_SECONDS = 60.0


def get_small_spec() -> BenchmarkSpec:
    """Return the canonical small autotune-oriented conv2d benchmark specification."""
    return BenchmarkSpec(
        name=f"small-conv2d-{TEST_H}x{TEST_W}",
        h=TEST_H,
        w=TEST_W,
        c_in=TEST_CIN,
        c_out=TEST_COUT,
        k=TEST_K,
        pad=TEST_PAD,
        ideal_seconds=SMALL_IDEAL_SECONDS,
        zero_score_seconds=SMALL_IDEAL_SECONDS * 10,
        stride=TEST_STRIDE,
    )


def get_mid_spec() -> BenchmarkSpec:
    """Return the canonical mid-sized conv2d benchmark specification."""
    return BenchmarkSpec(
        name=f"mid-conv2d-{MEDIUM_H}x{MEDIUM_W}",
        h=MEDIUM_H,
        w=MEDIUM_W,
        c_in=MEDIUM_CIN,
        c_out=MEDIUM_COUT,
        k=MEDIUM_K,
        pad=MEDIUM_PAD,
        ideal_seconds=MID_IDEAL_SECONDS,
        zero_score_seconds=MID_IDEAL_SECONDS * 10,
        stride=MEDIUM_STRIDE,
    )


def get_large_spec() -> BenchmarkSpec:
    """Return the canonical large runtime-oriented conv2d benchmark specification."""
    return BenchmarkSpec(
        name=f"large-conv2d-{RUNTIME_H}x{RUNTIME_W}",
        h=RUNTIME_H,
        w=RUNTIME_W,
        c_in=RUNTIME_CIN,
        c_out=RUNTIME_COUT,
        k=RUNTIME_K,
        pad=RUNTIME_PAD,
        ideal_seconds=LARGE_IDEAL_SECONDS,
        zero_score_seconds=LARGE_IDEAL_SECONDS * 10,
        stride=RUNTIME_STRIDE,
    )


def get_test_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical small conv2d benchmark."""
    return get_small_spec()


def get_medium_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical mid conv2d benchmark."""
    return get_mid_spec()


def get_runtime_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical large conv2d benchmark."""
    return get_large_spec()


def build_benchmark_spec(
    *,
    h=None,
    w=None,
    c_in=None,
    c_out=None,
    k=None,
    pad=None,
    stride=None,
    default_variant: str = "small",
    **kwargs,
) -> BenchmarkSpec:
    """Build a custom or named conv2d benchmark specification.

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
        The resolved conv2d benchmark specification.
    """
    del kwargs
    normalized_variant = (default_variant or "small").strip().lower()
    if normalized_variant == "medium":
        normalized_variant = "mid"
    if normalized_variant == "test":
        normalized_variant = "small"
    if normalized_variant == "runtime":
        normalized_variant = "large"

    if all(v is None for v in [h, w, c_in, c_out, k, pad, stride]):
        if normalized_variant == "mid":
            return get_mid_spec()
        if normalized_variant == "large":
            return get_large_spec()
        return get_small_spec()

    if normalized_variant == "mid":
        default_h, default_w = MEDIUM_H, MEDIUM_W
        default_c_in, default_c_out = MEDIUM_CIN, MEDIUM_COUT
        default_k, default_pad, default_stride = MEDIUM_K, MEDIUM_PAD, MEDIUM_STRIDE
        ideal_seconds = MID_IDEAL_SECONDS
    elif normalized_variant == "large":
        default_h, default_w = RUNTIME_H, RUNTIME_W
        default_c_in, default_c_out = RUNTIME_CIN, RUNTIME_COUT
        default_k, default_pad, default_stride = RUNTIME_K, RUNTIME_PAD, RUNTIME_STRIDE
        ideal_seconds = LARGE_IDEAL_SECONDS
    else:
        default_h, default_w = TEST_H, TEST_W
        default_c_in, default_c_out = TEST_CIN, TEST_COUT
        default_k, default_pad, default_stride = TEST_K, TEST_PAD, TEST_STRIDE
        ideal_seconds = SMALL_IDEAL_SECONDS

    return BenchmarkSpec(
        name="custom-conv2d",
        h=h if h is not None else default_h,
        w=w if w is not None else default_w,
        c_in=c_in if c_in is not None else default_c_in,
        c_out=c_out if c_out is not None else default_c_out,
        k=k if k is not None else default_k,
        pad=pad if pad is not None else default_pad,
        ideal_seconds=ideal_seconds,
        zero_score_seconds=ideal_seconds * 10,
        stride=stride if stride is not None else default_stride,
    )
