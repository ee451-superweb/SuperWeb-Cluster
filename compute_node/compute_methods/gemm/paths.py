"""Filesystem paths for the GEMM (cuBLAS) method runners."""

from __future__ import annotations

import os
from pathlib import Path

GEMM_METHOD_DIR = Path(__file__).resolve().parent

CUDA_DIR = GEMM_METHOD_DIR / "cuda"
CUDA_SOURCE_PATH = CUDA_DIR / "gemm_cuda_runner.cu"
CUDA_BUILD_DIR = CUDA_DIR / "build"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / (
    "gemm_cuda_runner.exe" if os.name == "nt" else "gemm_cuda_runner"
)
