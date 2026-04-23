"""Thin compute-node handler for the cuBLAS GEMM method.

Use this module when the compute node wants a simple handler wrapper around
the GEMM task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.compute_methods.gemm.executor import GemmTaskExecutor


class GemmMethodHandler:
    """Thin handler wrapper around the cuBLAS GEMM task executor."""

    method_name = "gemm"

    def __init__(self, *, pinned_backend: str | None = None) -> None:
        """Create the shared GEMM task executor for this handler.

        Args:
            pinned_backend: Optional backend name. GEMM is cuBLAS-only, so the
                only meaningful value is ``"cuda"``; any other non-None value
                is accepted but ignored because the executor never advertises
                another backend. Kept in the signature for parity with
                ``GemvMethodHandler`` and ``Conv2dMethodHandler``.
        """
        self._pinned_backend = pinned_backend
        self._executor = GemmTaskExecutor()

    def execute_task(self, task):
        """Execute one GEMM task through the shared executor.

        Args:
            task: GEMM task assignment to execute locally.

        Returns:
            The task result produced by the GEMM executor.
        """
        return self._executor.execute_task(task)

    def close(self) -> None:
        """Close resources owned by this handler, if any."""
        self._executor.close()
