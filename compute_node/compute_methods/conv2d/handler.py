"""Thin compute-node handler for the conv2d method.

Use this module when the compute node wants a simple handler wrapper around the
conv2d task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.compute_methods.conv2d.executor import Conv2dTaskExecutor


class Conv2dMethodHandler:
    """Thin handler wrapper around the conv2d task executor."""

    method_name = "conv2d"

    def __init__(self) -> None:
        """Create the shared conv2d task executor for this handler."""
        self._executor = Conv2dTaskExecutor()

    def execute_task(self, task):
        """Execute one conv2d task through the shared executor.

        Args:
            task: Conv2d task assignment to execute locally.

        Returns:
            The task result produced by the conv2d executor.
        """
        return self._executor.execute_task(task)

    def close(self) -> None:
        """Close resources owned by this handler, if any."""
        return None
