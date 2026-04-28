"""Thin compute-node handler for the GEMM method (cuBLAS + CPU fallback).

Use this module when the compute node wants a simple handler wrapper around
the GEMM task executor for registry-based dispatch.
"""

from __future__ import annotations

from compute_node.compute_methods.gemm.executor import GemmTaskExecutor


class GemmMethodHandler:
    """Thin handler wrapper around the GEMM task executor (cuBLAS or CPU)."""

    method_name = "gemm"

    def __init__(self, *, pinned_backend: str | None = None) -> None:
        """Create the shared GEMM task executor for this handler.

        Args:
            pinned_backend: Optional backend name. When set the executor is
                restricted to that backend's runner so a peer-cpu subprocess
                (which advertises CPU capacity) cannot accidentally dispatch
                to the cuBLAS runner just because the binary happens to exist
                on the shared filesystem of the parent host. Accepted values
                are ``"cpu"`` and ``"cuda"``; any other value is ignored.
        """
        self._pinned_backend = pinned_backend
        self._executor = GemmTaskExecutor(pinned_backend=pinned_backend)

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
