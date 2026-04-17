"""Hardware backend registry for the benchmark.

`benchmark.py` is intentionally hardware-agnostic. It should not know how to
compile MSVC code, how to invoke `nvcc`, or how to search CPU worker counts.

That hardware-specific logic lives under `backends/`.

Each backend object is responsible for one compute target, for example:

- CPU
- CUDA
- Metal
- OpenCL

Today the automatic benchmark flow wires in CPU, CUDA, and Metal. The DX12
module is retained on disk for postmortem debugging, but its benchmark entry
point is disabled because it can trigger fatal system instability on some
Windows machines.
"""

from __future__ import annotations

import os

from app.constants import DX12_BACKEND_DISABLED_REASON
from .cpu_backend import CpuBackend
from .cuda_backend import CudaBackend
from .metal_backend import MetalBackend
from .windows_gpu_inventory import detect_nvidia_windows_adapter


def _known_backend_factories() -> dict[str, type[object]]:
    """Return the backend classes that this workspace knows about."""

    return {
        "cpu": CpuBackend,
        "cuda": CudaBackend,
        "metal": MetalBackend,
    }


def _default_backend_names() -> list[str]:
    """Pick the backends that should run automatically.

    On Windows we keep the default path conservative and exclude DX12 entirely
    because the DX12 module is currently considered unsafe to run.
    """

    names = ["cpu"]

    if os.name == "nt":
        nvidia_adapter_name, _ = detect_nvidia_windows_adapter()
        if nvidia_adapter_name is not None:
            names.append("cuda")
        return names

    names.extend(["cuda", "metal"])
    return names


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
        if name == "dx12":
            raise ValueError(DX12_BACKEND_DISABLED_REASON)
        if name not in factories:
            raise ValueError(f"unknown backend {name!r}")
        backends.append(factories[name]())
    return backends
