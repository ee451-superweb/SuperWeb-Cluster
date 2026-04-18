"""Thin compute-node handler for the fixed-matrix-vector method.

Use this module when the compute node wants a simple handler wrapper around the
GEMV task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.task_executor import GemvTaskExecutor


class GemvMethodHandler:
    """Thin handler wrapper around the legacy GEMV task executor."""

    method_name = "gemv"

    def __init__(self, inventory=None) -> None:
        """Create the shared GEMV task executor for this handler.

        Args:
            inventory: Optional runtime processor inventory override.
        """
        self._executor = GemvTaskExecutor(inventory)

    def execute_task(self, task):
        """Execute one GEMV task through the shared executor.

        Args:
            task: GEMV task assignment to execute locally.

        Returns:
            The task result produced by the GEMV executor.
        """
        return self._executor.execute_task(task)

    def close(self) -> None:
        """Close resources owned by the GEMV task executor."""
        self._executor.close()
