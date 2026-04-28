"""Thin compute-node handler for the fixed-matrix-vector method.

Use this module when the compute node wants a simple handler wrapper around the
GEMV task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.task_executor import GemvTaskExecutor


class GemvMethodHandler:
    """Thin handler wrapper around the legacy GEMV task executor."""

    method_name = "gemv"

    def __init__(self, inventory=None, *, pinned_backend: str | None = None) -> None:
        """Create the shared GEMV task executor for this handler.

        Args:
            inventory: Optional runtime processor inventory override.
            pinned_backend: Optional backend name passed to the executor so it
                only advertises and uses the pinned backend at runtime.
        """
        self._executor = GemvTaskExecutor(inventory, pinned_backend=pinned_backend)

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
