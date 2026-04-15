"""Compute-node handler for the spatial-convolution method."""

from __future__ import annotations

from compute_node.compute_methods.spatial_convolution.executor import SpatialConvolutionTaskExecutor


class SpatialConvolutionMethodHandler:
    """Thin handler wrapper around the spatial-convolution task executor."""

    method_name = "spatial_convolution"

    def __init__(self) -> None:
        self._executor = SpatialConvolutionTaskExecutor()

    def execute_task(self, task):
        return self._executor.execute_task(task)

    def close(self) -> None:
        return None
