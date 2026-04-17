"""Thin compute-node handler for the fixed-matrix-vector method.

Use this module when the compute node wants a simple handler wrapper around the
FMVM task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.task_executor import FixedMatrixVectorTaskExecutor


class FixedMatrixVectorMethodHandler:
    """Thin handler wrapper around the legacy FMVM task executor."""

    method_name = "fixed_matrix_vector_multiplication"

    def __init__(self, inventory=None) -> None:
        """Create the shared FMVM task executor for this handler.

        Args:
            inventory: Optional runtime processor inventory override.
        """
        self._executor = FixedMatrixVectorTaskExecutor(inventory)

    def execute_task(self, task):
        """Execute one FMVM task through the shared executor.

        Args:
            task: FMVM task assignment to execute locally.

        Returns:
            The task result produced by the FMVM executor.
        """
        return self._executor.execute_task(task)

    def close(self) -> None:
        """Close resources owned by the FMVM task executor."""
        self._executor.close()
