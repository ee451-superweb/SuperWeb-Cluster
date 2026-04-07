"""Backend registry for performance benchmarks."""

from __future__ import annotations

from backends.cpu_backend import CpuBackend
from backends.cuda_backend import CudaBackend


def build_backends(names: list[str]) -> list[object]:
    """Instantiate the requested backends."""

    normalized = []
    for name in names:
        if name == "all":
            normalized.extend(["cpu", "cuda"])
        else:
            normalized.append(name)

    ordered: list[str] = []
    for name in normalized:
        if name not in ordered:
            ordered.append(name)

    backends: list[object] = []
    for name in ordered:
        if name == "cpu":
            backends.append(CpuBackend())
        elif name == "cuda":
            backends.append(CudaBackend())
        else:
            raise ValueError(f"unknown backend {name!r}")

    return backends
