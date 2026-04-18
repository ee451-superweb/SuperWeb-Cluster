"""Filesystem paths for the shared fixed matrix-vector method runners."""

from __future__ import annotations

import os
import sys
from pathlib import Path

GEMV_METHOD_DIR = Path(__file__).resolve().parent

CPU_DIR = GEMV_METHOD_DIR / "cpu"
CPU_WINDOWS_DIR = CPU_DIR / "windows"
CPU_WINDOWS_SOURCE_PATH = CPU_WINDOWS_DIR / "gemv_cpu_windows.cpp"
CPU_WINDOWS_BUILD_DIR = CPU_WINDOWS_DIR / "build"
CPU_WINDOWS_EXECUTABLE_PATH = CPU_WINDOWS_BUILD_DIR / "gemv_cpu_windows.exe"

CPU_MACOS_DIR = CPU_DIR / "macos"
CPU_MACOS_SOURCE_PATH = CPU_MACOS_DIR / "gemv_cpu_macos.cpp"
CPU_MACOS_BUILD_DIR = CPU_MACOS_DIR / "build"
CPU_MACOS_EXECUTABLE_PATH = CPU_MACOS_BUILD_DIR / "gemv_cpu_macos"

CUDA_DIR = GEMV_METHOD_DIR / "cuda"
CUDA_SOURCE_PATH = CUDA_DIR / "gemv_cuda_runner.cu"
CUDA_BUILD_DIR = CUDA_DIR / "build"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / ("gemv_cuda_runner.exe" if os.name == "nt" else "gemv_cuda_runner")

DX12_DIR = GEMV_METHOD_DIR / "dx12"
DX12_SOURCE_PATH = DX12_DIR / "gemv_dx12_runner.cpp"
DX12_BUILD_DIR = DX12_DIR / "build"
DX12_EXECUTABLE_PATH = DX12_BUILD_DIR / "gemv_dx12_runner.exe"

METAL_DIR = GEMV_METHOD_DIR / "metal"
METAL_HOST_SOURCE_PATH = METAL_DIR / "gemv_metal_runner.mm"
METAL_KERNEL_SOURCE_PATH = METAL_DIR / "gemv_metal_kernels.metal"
METAL_BUILD_DIR = METAL_DIR / "build"
METAL_EXECUTABLE_PATH = METAL_BUILD_DIR / "gemv_metal_runner"
METAL_AIR_PATH = METAL_BUILD_DIR / "gemv_metal_kernels.air"
METAL_LIBRARY_PATH = METAL_BUILD_DIR / "gemv_metal_kernels.metallib"


def current_cpu_executable_path() -> Path:
    """Return the checked-in CPU runner that matches the current host OS."""

    if sys.platform == "win32":
        return CPU_WINDOWS_EXECUTABLE_PATH
    if sys.platform == "darwin":
        return CPU_MACOS_EXECUTABLE_PATH
    raise RuntimeError(f"gemv CPU runner is unsupported on {sys.platform!r}")
