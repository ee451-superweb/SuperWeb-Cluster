"""Generic task router for compute-node runtime execution."""

from __future__ import annotations

from compute_node.handlers import MethodHandlerRegistry


class TaskExecutionRouter:
    """Route each assigned task to the method-specific local handler."""

    def __init__(self, handler_registry: MethodHandlerRegistry) -> None:
        self._handler_registry = handler_registry

    def execute_task(self, task):
        return self._handler_registry.get(task.method).execute_task(task)

    def close(self) -> None:
        self._handler_registry.close_all()
