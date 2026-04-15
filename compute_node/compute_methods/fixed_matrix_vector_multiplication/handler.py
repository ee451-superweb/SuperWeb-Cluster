"""Compute-node handler for the fixed-matrix-vector method."""

from __future__ import annotations

from compute_node.task_executor import FixedMatrixVectorTaskExecutor


class FixedMatrixVectorMethodHandler:
    """Thin handler wrapper around the legacy FMVM task executor."""

    method_name = "fixed_matrix_vector_multiplication"

    def __init__(self, inventory=None) -> None:
        self._executor = FixedMatrixVectorTaskExecutor(inventory)

    def execute_task(self, task):
        return self._executor.execute_task(task)

    def close(self) -> None:
        self._executor.close()
