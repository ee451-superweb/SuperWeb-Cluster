"""Thin compute-node handler for the spatial-convolution method.

Use this module when the compute node wants a simple handler wrapper around the
spatial task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.compute_methods.spatial_convolution.executor import SpatialConvolutionTaskExecutor


class SpatialConvolutionMethodHandler:
    """Thin handler wrapper around the spatial-convolution task executor."""

    method_name = "spatial_convolution"

    def __init__(self) -> None:
        """Create the shared spatial task executor for this handler."""
        self._executor = SpatialConvolutionTaskExecutor()

    def execute_task(self, task):
        """Execute one spatial task through the shared executor.

        Args:
            task: Spatial task assignment to execute locally.

        Returns:
            The task result produced by the spatial executor.
        """
        return self._executor.execute_task(task)

    def close(self) -> None:
        """Close resources owned by this handler, if any."""
        return None
