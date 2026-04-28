"""Filesystem paths for the GEMM method runners (CPU + cuBLAS)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

GEMM_METHOD_DIR = Path(__file__).resolve().parent

CPU_DIR = GEMM_METHOD_DIR / "cpu"
CPU_WINDOWS_DIR = CPU_DIR / "windows"
CPU_WINDOWS_SOURCE_PATH = CPU_WINDOWS_DIR / "gemm_cpu_windows.cpp"
CPU_WINDOWS_BUILD_DIR = CPU_WINDOWS_DIR / "build"
CPU_WINDOWS_EXECUTABLE_PATH = CPU_WINDOWS_BUILD_DIR / "gemm_cpu_windows.exe"

CPU_MACOS_DIR = CPU_DIR / "macos"
CPU_MACOS_SOURCE_PATH = CPU_MACOS_DIR / "gemm_cpu_macos.cpp"
CPU_MACOS_BUILD_DIR = CPU_MACOS_DIR / "build"
CPU_MACOS_EXECUTABLE_PATH = CPU_MACOS_BUILD_DIR / "gemm_cpu_macos"

CUDA_DIR = GEMM_METHOD_DIR / "cuda"
CUDA_SOURCE_PATH = CUDA_DIR / "gemm_cuda_runner.cu"
CUDA_BUILD_DIR = CUDA_DIR / "build"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / (
    "gemm_cuda_runner.exe" if os.name == "nt" else "gemm_cuda_runner"
)


def current_cpu_executable_path() -> Path:
    """Return the checked-in CPU GEMM runner that matches the current host OS.

    Use this when selecting a runner at runtime: Windows ships the ``.exe``
    built under ``cpu/windows/build``, macOS ships the bare executable under
    ``cpu/macos/build``. Linux is not yet provided as a first-class backend;
    callers can still point at the macOS source and rebuild with g++.
    """

    if sys.platform == "win32":
        return CPU_WINDOWS_EXECUTABLE_PATH
    if sys.platform == "darwin":
        return CPU_MACOS_EXECUTABLE_PATH
    raise RuntimeError(f"gemm CPU runner is unsupported on {sys.platform!r}")


def current_cpu_source_path() -> Path:
    """Return the CPU GEMM runner source file that matches the current host OS."""

    if sys.platform == "win32":
        return CPU_WINDOWS_SOURCE_PATH
    if sys.platform == "darwin":
        return CPU_MACOS_SOURCE_PATH
    raise RuntimeError(f"gemm CPU runner is unsupported on {sys.platform!r}")
