"""Benchmark constants for the Convolution workload."""

from __future__ import annotations

from compute_node.compute_methods.spatial_convolution.performance_metrics.models import BenchmarkSpec
from compute_node.input_matrix.spatial_convolution.spec import (
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
    return BenchmarkSpec(
        name=f"test-conv2d-{TEST_H}x{TEST_W}",
        h=TEST_H, w=TEST_W, c_in=TEST_CIN, c_out=TEST_COUT, k=TEST_K, pad=TEST_PAD,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10, stride=TEST_STRIDE,
    )

def get_runtime_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"runtime-conv2d-{RUNTIME_H}x{RUNTIME_W}",
        h=RUNTIME_H, w=RUNTIME_W, c_in=RUNTIME_CIN, c_out=RUNTIME_COUT, k=RUNTIME_K, pad=RUNTIME_PAD,
        ideal_seconds=RUNTIME_IDEAL_SECONDS, zero_score_seconds=RUNTIME_IDEAL_SECONDS * 10, stride=RUNTIME_STRIDE,
    )

def build_benchmark_spec(*, h=None, w=None, c_in=None, c_out=None, k=None, pad=None, stride=None, **kwargs) -> BenchmarkSpec:
    if all(v is None for v in [h, w, c_in, c_out, k, pad, stride]):
        return get_test_spec()
    return BenchmarkSpec(
        name="custom-conv2d",
        h=h if h is not None else TEST_H, 
        w=w if w is not None else TEST_W, 
        c_in=c_in if c_in is not None else TEST_CIN, 
        c_out=c_out if c_out is not None else TEST_COUT,
        k=k if k is not None else TEST_K, 
        pad=pad if pad is not None else TEST_PAD,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10,
        stride=stride if stride is not None else TEST_STRIDE,
    )
