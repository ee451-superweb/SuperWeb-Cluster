"""Windows CPU backend for the fixed matrix-vector benchmark.

This file exists to bridge two worlds:

- Python orchestration in `benchmark.py`
- the hardware-specific C++ runner in
  `fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`

The important design choice is that the C++ program reads `A.bin` and `x.bin`
only once, then searches multiple worker/tile configurations internally. That
keeps the benchmark focused on compute performance instead of repeatedly paying
the 2 GiB input-file I/O cost.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from models import BackendResult, BenchmarkSpec, DatasetLayout, TrialRecord
from path_utils import sanitize_text, to_relative_cli_path, to_relative_string
from scoring import linear_time_score

ROOT_DIR = Path(__file__).resolve().parents[1]
CPU_WINDOWS_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "cpu" / "windows"
CPU_SOURCE_PATH = CPU_WINDOWS_DIR / "fmvm_cpu_windows.cpp"
CPU_BUILD_DIR = CPU_WINDOWS_DIR / "build"
CPU_EXECUTABLE_PATH = CPU_BUILD_DIR / "fmvm_cpu_windows.exe"


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


def _binary_tree_worker_candidates(hardware_workers: int) -> list[int]:
    """Return the worker counts requested by the user.

    For example, on a 16-core machine this returns:

    - 16
    - 8
    - 32
    - 4
    - 64
    """

    root = max(1, hardware_workers)
    candidates = [
        root,
        max(1, root // 2),
        root * 2,
        max(1, root // 4),
        root * 4,
    ]

    ordered: list[int] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _candidate_tile_sizes(cols: int) -> list[int]:
    """Return the tile sizes the CPU runner should sweep.

    We keep the list short because the default benchmark already uses a 2 GiB
    matrix, so each extra configuration has a noticeable runtime cost.
    """

    candidates: list[int] = []
    for value in (256, 512, 1024, 2048, 4096, 8192):
        if value <= cols:
            candidates.append(value)
    if not candidates:
        candidates.append(cols)
    if cols not in candidates:
        candidates.append(cols)
    return candidates


def _default_repeats(spec: BenchmarkSpec) -> int:
    """Choose how many timed repeats each configuration should run.

    Large datasets need fewer repeats or the full search would take too long.
    Small test datasets can afford more repeats to smooth out timer noise.
    """

    gib = 1024**3
    if spec.matrix_bytes >= 2 * gib:
        return 1
    if spec.matrix_bytes >= 256 * 1024**2:
        return 2
    return 6


class CpuBackend:
    """Compile and invoke the Windows C++ CPU runner."""

    name = "cpu"

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine can build and run the Windows CPU backend."""

        if sys.platform != "win32":
            return False, "Windows C++ CPU backend is only available on Windows."
        if not CPU_SOURCE_PATH.exists():
            return False, f"missing CPU runner source at {_relative_project_path(CPU_SOURCE_PATH)}"

        vsdevcmd_path = self._find_vsdevcmd()
        if vsdevcmd_path is None:
            return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."

        worker_candidates = _binary_tree_worker_candidates(os.cpu_count() or 1)
        return (
            True,
            f"Windows CPU backend available. worker search order: {worker_candidates}",
        )

    def run(
        self,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        time_budget_seconds: float,
    ) -> BackendResult:
        """Run the C++ CPU executable once and return its best found configuration."""

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
            notes.append(_sanitize_note(f"failed to compile CPU runner: {exc}"))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        worker_candidates = _binary_tree_worker_candidates(os.cpu_count() or 1)
        tile_candidates = _candidate_tile_sizes(spec.cols)
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
            "--workers",
            ",".join(str(value) for value in worker_candidates),
            "--tile-sizes",
            ",".join(str(value) for value in tile_candidates),
            "--repeats",
            str(repeats),
        ]

        notes.append(f"tile search order: {tile_candidates}")
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
            notes.append("CPU benchmark timed out")
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
            notes.append(_sanitize_note(stderr if stderr else str(exc)))
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
            "workers": int(metrics["actual_workers"]),
            "requested_workers": int(metrics["requested_workers"]),
            "tile_size": int(metrics["tile_size"]),
            "repeats": int(metrics["repeats"]),
            "trials_run": int(metrics["trials_run"]),
        }
        trial = TrialRecord(
            backend=self.name,
            config=config,
            wall_clock_latency_seconds=float(metrics["wall_clock_latency_seconds"]),
            effective_gflops=float(metrics["effective_gflops"]),
            checksum=str(metrics["checksum"]),
            score=score,
            notes=[],
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
        """Build the Windows CPU runner when the source changed."""

        CPU_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        if CPU_EXECUTABLE_PATH.exists() and CPU_EXECUTABLE_PATH.stat().st_mtime >= CPU_SOURCE_PATH.stat().st_mtime:
            return CPU_EXECUTABLE_PATH

        vsdevcmd_path = self._find_vsdevcmd()
        if vsdevcmd_path is None:
            raise FileNotFoundError("VsDevCmd.bat was not found")

        compile_script_path = CPU_BUILD_DIR / "build_fmvm_cpu_windows.cmd"
        compile_script_path.write_text(
            "\n".join(
                [
                    "@echo off",
                    *_windows_vsdevcmd_setup_lines(),
                    "pushd \"%~dp0\"",
                    # `/fp:fast` gives the compiler freedom to reassociate the
                    # FP32 reduction, which is appropriate for this performance
                    # benchmark because correctness is already checked with
                    # tolerance-based validation rather than exact bit matches.
                    "cl /nologo /std:c++20 /O2 /fp:fast /EHsc /Fe:fmvm_cpu_windows.exe ..\\fmvm_cpu_windows.cpp",
                    "set \"BUILD_EXIT=%ERRORLEVEL%\"",
                    "popd",
                    "exit /b %BUILD_EXIT%",
                ]
            )
            + "\n",
            encoding="ascii",
        )

        # PowerShell's call operator handles a batch-file path with spaces more
        # reliably than invoking cmd.exe directly through subprocess argument
        # quoting. This matters because "performance metrics" is part of the
        # workspace path.
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
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return CPU_EXECUTABLE_PATH

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
