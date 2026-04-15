"""Shared paths and helpers for the spatial-convolution compute method."""

from .executor import (
    DATASET_GENERATE_SCRIPT_PATH,
    DEFAULT_CONV_RESULT_PATH,
    DEFAULT_DATASET_DIR,
    PERF_DIR,
    SpatialConvolutionTaskExecutor,
)

__all__ = [
    "DATASET_GENERATE_SCRIPT_PATH",
    "DEFAULT_CONV_RESULT_PATH",
    "DEFAULT_DATASET_DIR",
    "PERF_DIR",
    "SpatialConvolutionTaskExecutor",
]
