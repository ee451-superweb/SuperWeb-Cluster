"""Method-handler registry for compute-node task execution."""

from __future__ import annotations

from dataclasses import dataclass

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from compute_node.compute_methods.fixed_matrix_vector_multiplication.handler import FixedMatrixVectorMethodHandler
from compute_node.compute_methods.spatial_convolution.handler import SpatialConvolutionMethodHandler


@dataclass(slots=True)
class MethodHandlerRegistry:
    """Lookup table from method name to one local compute handler."""

    handlers: dict[str, object]

    def get(self, method: str):
        handler = self.handlers.get(method)
        if handler is None:
            raise ValueError(f"unsupported task method: {method}")
        return handler

    def close_all(self) -> None:
        for handler in self.handlers.values():
            close = getattr(handler, "close", None)
            if callable(close):
                close()


def build_default_method_handlers() -> MethodHandlerRegistry:
    """Create the default compute-node method handlers."""

    return MethodHandlerRegistry(
        handlers={
            METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION: FixedMatrixVectorMethodHandler(),
            METHOD_SPATIAL_CONVOLUTION: SpatialConvolutionMethodHandler(),
        }
    )
