"""Hardware backend registry for the benchmark.

`benchmark.py` is intentionally hardware-agnostic. It should not know how to
compile MSVC code, how to invoke `nvcc`, or how to search CPU worker counts.

That hardware-specific logic lives under `backends/`.

Each backend object is responsible for one compute target, for example:

- CPU
- CUDA
- Metal
- OpenCL

Today the automatic benchmark flow wires in CPU, CUDA, and Metal, and this
folder is where future hardware-specific runners should be added.
"""

from __future__ import annotations

from .cpu_backend import CpuBackend
from .cuda_backend import CudaBackend
from .metal_backend import MetalBackend


def _known_backend_factories() -> dict[str, type[object]]:
    """Return the backend classes that this workspace knows about."""

    return {
        "cpu": CpuBackend,
        "cuda": CudaBackend,
        "metal": MetalBackend,
    }


def _default_backend_names() -> list[str]:
    """Pick the backends that should run automatically.

    The benchmark should try every backend we currently know how to orchestrate.
    Availability checks happen inside the backend objects themselves, so it is
    safe to include CUDA here even on machines without a CUDA toolchain.
    """

    return ["cpu", "cuda", "metal"]


def build_backends(names: list[str] | None = None) -> list[object]:
    """Instantiate the requested backends.

    If the caller does not request anything explicitly, we use the current
    default backend order.
    """

    requested_names = _default_backend_names() if not names else names
    factories = _known_backend_factories()

    ordered_names: list[str] = []
    for name in requested_names:
        if name == "all":
            for default_name in _default_backend_names():
                if default_name not in ordered_names:
                    ordered_names.append(default_name)
            continue

        if name not in ordered_names:
            ordered_names.append(name)

    backends: list[object] = []
    for name in ordered_names:
        if name not in factories:
            raise ValueError(f"unknown backend {name!r}")
        backends.append(factories[name]())
    return backends
