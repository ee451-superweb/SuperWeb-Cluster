"""CUDA backend for the fixed matrix-vector benchmark.

This backend mirrors the CPU design:

- Python handles orchestration, compilation, and result parsing
- the CUDA executable handles loading the dataset once and sweeping multiple
  kernel configurations in memory

That split keeps the expensive dataset I/O out of the tuning loop and makes the
top-level benchmark code stay hardware-agnostic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from models import BackendResult, BenchmarkSpec, DatasetLayout, TrialRecord
from path_utils import sanitize_text, to_relative_cli_path, to_relative_string
from scoring import linear_time_score

ROOT_DIR = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "cuda"
CUDA_SOURCE_PATH = CUDA_DIR / "fmvm_cuda_runner.cu"
CUDA_BUILD_DIR = CUDA_DIR / "build"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / ("fmvm_cuda_runner.exe" if os.name == "nt" else "fmvm_cuda_runner")


def _relative_project_path(path: Path) -> str:
    """Show a project-local path without an absolute prefix."""

    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Remove project/home absolute prefixes from backend notes."""

    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render a relative path for subprocess calls."""

    return to_relative_cli_path(path, start=ROOT_DIR)


def _windows_vsdevcmd_setup_lines() -> list[str]:
    """Emit batch-script lines that locate VsDevCmd without project absolute paths."""

    return [
        "setlocal",
        "set \"VSDEVCMD=\"",
        "if exist \"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" (",
        "  for /f \"usebackq delims=\" %%i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" -latest -products * -find Common7\\Tools\\VsDevCmd.bat`) do set \"VSDEVCMD=%%i\"",
        ")",
        "if not defined VSDEVCMD set \"VSDEVCMD=%ProgramFiles(x86)%\\Microsoft Visual Studio\\18\\BuildTools\\Common7\\Tools\\VsDevCmd.bat\"",
        "if not exist \"%VSDEVCMD%\" exit /b 1",
        "call \"%VSDEVCMD%\" -arch=x64 -host_arch=x64 >nul",
    ]


def _candidate_block_sizes() -> list[int]:
    """Return the CUDA block sizes to sweep."""

    return [64, 128, 256, 512]


def _candidate_tile_sizes() -> list[int]:
    """Return the template tile sizes supported by the CUDA kernels."""

    return [1, 2, 4, 8]


def _candidate_transpose_modes() -> list[int]:
    """Try both the row-major input layout and a transposed GPU layout."""

    return [0, 1]


def _default_repeats(spec: BenchmarkSpec) -> int:
    """Choose a repeat count that keeps large benchmarks short."""

    gib = 1024**3
    if spec.matrix_bytes >= 2 * gib:
        return 2
    if spec.matrix_bytes >= 256 * 1024**2:
        return 4
    return 8


def _detect_compute_capability() -> str | None:
    """Ask `nvidia-smi` for the first GPU's compute capability."""

    if shutil.which("nvidia-smi") is None:
        return None

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digits = stripped.replace(".", "")
        if digits.isdigit():
            return digits
    return None


class CudaBackend:
    """Compile and invoke the CUDA autotuning runner."""

    name = "cuda"

    def probe(self) -> tuple[bool, str]:
        """Check whether the current machine looks CUDA-capable."""

        if shutil.which("nvcc") is None:
            return False, "nvcc was not found on PATH; CUDA backend is unavailable."
        if not CUDA_SOURCE_PATH.exists():
            return False, f"missing CUDA runner source at {_relative_project_path(CUDA_SOURCE_PATH)}"

        capability = _detect_compute_capability()
        if capability is None:
            return True, "nvcc detected on PATH, but compute capability could not be queried."
        return True, f"nvcc detected on PATH; first GPU compute capability is sm_{capability}."

    def run(
        self,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        time_budget_seconds: float,
    ) -> BackendResult:
        """Run the CUDA executable once and return the best configuration it found."""

        available, message = self.probe()
        notes = [message]
        if not available:
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        try:
            executable_path = self._compile_if_needed()
        except (OSError, subprocess.CalledProcessError) as exc:
            details = ""
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or exc.stdout or "").strip()
            notes.append(_sanitize_note(f"failed to compile CUDA runner: {details or exc}"))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        block_sizes = _candidate_block_sizes()
        tile_sizes = _candidate_tile_sizes()
        transpose_modes = _candidate_transpose_modes()
        repeats = _default_repeats(spec)
        command = [
            str(executable_path),
            "--matrix",
            _relative_cli_path(dataset.matrix_path),
            "--vector",
            _relative_cli_path(dataset.vector_path),
            "--rows",
            str(spec.rows),
            "--cols",
            str(spec.cols),
            "--transpose-modes",
            ",".join(str(value) for value in transpose_modes),
            "--block-sizes",
            ",".join(str(value) for value in block_sizes),
            "--tile-sizes",
            ",".join(str(value) for value in tile_sizes),
            "--repeats",
            str(repeats),
        ]

        notes.append(f"transpose search order: {transpose_modes}")
        notes.append(f"block size search order: {block_sizes}")
        notes.append(f"tile size search order: {tile_sizes}")
        notes.append(f"repeats_per_config: {repeats}")

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(time_budget_seconds, 30.0),
                cwd=ROOT_DIR,
            )
        except subprocess.TimeoutExpired:
            notes.append("CUDA benchmark timed out")
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            notes.append(_sanitize_note(stderr if stderr else (stdout if stdout else str(exc))))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        metrics = json.loads(completed.stdout)
        score = linear_time_score(
            float(metrics["wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds,
        )

        config = {
            "transpose": bool(metrics["transpose"]),
            "block_size": int(metrics["block_size"]),
            "tile_size": int(metrics["tile_size"]),
            "repeats": int(metrics["repeats"]),
            "trials_run": int(metrics["trials_run"]),
        }
        trial_notes: list[str] = []
        if "device_name" in metrics:
            trial_notes.append(f"device={metrics['device_name']}")
        if "compute_capability" in metrics:
            trial_notes.append(f"sm={metrics['compute_capability']}")

        trial = TrialRecord(
            backend=self.name,
            config=config,
            wall_clock_latency_seconds=float(metrics["wall_clock_latency_seconds"]),
            effective_gflops=float(metrics["effective_gflops"]),
            checksum=str(metrics["checksum"]),
            score=score,
            notes=trial_notes,
        )
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=dict(config),
            best_trial=trial,
            trials=[trial],
            notes=notes,
        )

    def _compile_if_needed(self) -> Path:
        """Build the CUDA runner when the source changed."""

        CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        if CUDA_EXECUTABLE_PATH.exists() and CUDA_EXECUTABLE_PATH.stat().st_mtime >= CUDA_SOURCE_PATH.stat().st_mtime:
            return CUDA_EXECUTABLE_PATH

        if shutil.which("nvcc") is None:
            raise FileNotFoundError("nvcc was not found")

        capability = _detect_compute_capability()
        if os.name == "nt":
            vsdevcmd_path = self._find_vsdevcmd()
            if vsdevcmd_path is None:
                raise FileNotFoundError("VsDevCmd.bat was not found for CUDA compilation")

            command_parts = [
                "nvcc",
                "..\\fmvm_cuda_runner.cu",
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-o fmvm_cuda_runner.exe",
            ]
            if capability is not None:
                command_parts.append(f"-gencode=arch=compute_{capability},code=sm_{capability}")

            compile_script_path = CUDA_BUILD_DIR / "build_fmvm_cuda_runner.cmd"
            compile_script_path.write_text(
                "\n".join(
                    [
                        "@echo off",
                        *_windows_vsdevcmd_setup_lines(),
                        "pushd \"%~dp0\"",
                        " ".join(command_parts),
                        "set \"BUILD_EXIT=%ERRORLEVEL%\"",
                        "popd",
                        "exit /b %BUILD_EXIT%",
                    ]
                )
                + "\n",
                encoding="ascii",
            )

            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"& '{compile_script_path}'",
                ],
                capture_output=True,
                text=True,
            )
        else:
            command = [
                "nvcc",
                "../fmvm_cuda_runner.cu",
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-o",
                CUDA_EXECUTABLE_PATH.name,
            ]
            if capability is not None:
                command.append(f"-gencode=arch=compute_{capability},code=sm_{capability}")
            completed = subprocess.run(command, capture_output=True, text=True, cwd=CUDA_BUILD_DIR)

        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return CUDA_EXECUTABLE_PATH

    @staticmethod
    def _find_vsdevcmd() -> Path | None:
        """Resolve the Visual Studio developer-command batch file."""

        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if not program_files_x86:
            return None

        vswhere_path = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere_path.exists():
            completed = subprocess.run(
                [
                    str(vswhere_path),
                    "-latest",
                    "-products",
                    "*",
                    "-find",
                    "Common7\\Tools\\VsDevCmd.bat",
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                resolved = completed.stdout.strip().splitlines()
                if resolved:
                    candidate = Path(resolved[0].strip())
                    if candidate.exists():
                        return candidate

        fallback_path = (
            Path(program_files_x86)
            / "Microsoft Visual Studio"
            / "18"
            / "BuildTools"
            / "Common7"
            / "Tools"
            / "VsDevCmd.bat"
        )
        if fallback_path.exists():
            return fallback_path
        return None
