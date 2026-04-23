"""Route compute-node tasks to their method-specific executors.

Use this module when the compute node wants one thin abstraction above the
method-handler registry for local task execution and shutdown.
"""

from __future__ import annotations

from compute_node.worker_handlers import MethodHandlerRegistry


class TaskExecutionRouter:
    """Route each assigned task to the method-specific local handler."""

    def __init__(self, handler_registry: MethodHandlerRegistry) -> None:
        """Store the method-handler registry used for local task execution.

        Args:
            handler_registry: Registry that maps method names to local handlers.
        """
        self._handler_registry = handler_registry

    def execute_task(self, task):
        """Dispatch one task to its method-specific local handler.

        Args:
            task: Runtime task assignment to execute locally.

        Returns:
            The task result produced by the selected method handler.
        """
        return self._handler_registry.get(task.method).execute_task(task)

    def close(self) -> None:
        """Close all registered method handlers owned by this router."""
        self._handler_registry.close_all()
