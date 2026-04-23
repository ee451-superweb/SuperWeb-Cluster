"""Build and manage method-handler objects for compute-node task execution.

Use this module when the compute node needs a stable lookup table from method
name to the local executor that can handle that task type.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import METHOD_GEMM, METHOD_GEMV, METHOD_CONV2D
from compute_node.compute_methods.gemm.handler import GemmMethodHandler
from compute_node.compute_methods.gemv.handler import GemvMethodHandler
from compute_node.compute_methods.conv2d.handler import Conv2dMethodHandler


@dataclass(slots=True)
class MethodHandlerRegistry:
    """Lookup table from method name to one local compute handler."""

    handlers: dict[str, object]

    def get(self, method: str):
        """Return the registered handler for one method name.

        Args:
            method: Logical task method name.

        Returns:
            The handler object registered for that method.
        """
        handler = self.handlers.get(method)
        if handler is None:
            raise ValueError(f"unsupported task method: {method}")
        return handler

    def close_all(self) -> None:
        """Call ``close()`` on every registered handler that supports it."""
        for handler in self.handlers.values():
            close = getattr(handler, "close", None)
            if callable(close):
                close()


def build_default_method_handlers(
    *,
    pinned_backend: str | None = None,
) -> MethodHandlerRegistry:
    """Create the default compute-node method handlers.

    Args:
        pinned_backend: Optional backend name restricting every handler's
            inventory so a dual-purpose peer only executes its pinned backend.

    Returns:
        A method-handler registry populated with the built-in handlers.
    """

    return MethodHandlerRegistry(
        handlers={
            METHOD_GEMV: GemvMethodHandler(pinned_backend=pinned_backend),
            METHOD_CONV2D: Conv2dMethodHandler(pinned_backend=pinned_backend),
            METHOD_GEMM: GemmMethodHandler(pinned_backend=pinned_backend),
        }
    )
