"""Windows C++ CPU backend for fixed matrix-vector multiplication."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fmvm_dataset import compare_vectors, load_float32_file
from models import BackendResult, DatasetPaths, TrialRecord, WorkloadSpec
from scoring import linear_time_score

ROOT_DIR = Path(__file__).resolve().parents[1]
CPU_WINDOWS_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "cpu" / "windows"
CPU_SOURCE_PATH = CPU_WINDOWS_DIR / "fmvm_cpu_windows.cpp"
CPU_BUILD_DIR = CPU_WINDOWS_DIR / "build"
CPU_EXECUTABLE_PATH = CPU_BUILD_DIR / "fmvm_cpu_windows.exe"
CPU_OUTPUT_PATH = CPU_BUILD_DIR / "fmvm_cpu_output.bin"


def _binary_tree_worker_candidates(hardware_workers: int) -> list[int]:
    """Return the requested worker order rooted at the hardware worker count."""

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


def _candidate_tile_sizes(limit: int) -> list[int]:
    """Return the tile sizes swept for every worker-count candidate."""

    values = []
    for value in (32, 64, 128, 256, 512, 1024):
        if value <= limit:
            values.append(value)
    if not values:
        values.append(limit)
    if limit not in values:
        values.append(limit)
    return values


def _default_repeats(workload: WorkloadSpec) -> int:
    """Keep short workloads stable without making the overall benchmark drag."""

    if workload.preset == "smoke":
        return 16
    if workload.preset == "quick":
        return 12
    if workload.preset == "standard":
        return 8
    return 4


class CpuBackend:
    """CPU backend that compiles and invokes the Windows C++ runner."""

    name = "cpu"

    def probe(self) -> tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Windows C++ CPU backend is only available on Windows."
        if not CPU_SOURCE_PATH.exists():
            return False, f"missing CPU runner source at {CPU_SOURCE_PATH}"

        vsdevcmd_path = self._find_vsdevcmd()
        if vsdevcmd_path is None:
            return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."

        hardware_workers = os.cpu_count() or 1
        worker_candidates = _binary_tree_worker_candidates(hardware_workers)
        return (
            True,
            "Windows C++ CPU backend available via "
            f"{vsdevcmd_path}. worker search order: {worker_candidates}",
        )

    def run(
        self,
        workload: WorkloadSpec,
        dataset: DatasetPaths,
        reference_output: list[float],
        *,
        time_budget_seconds: float,
    ) -> BackendResult:
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
            notes.append(f"failed to compile CPU runner: {exc}")
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        hardware_workers = os.cpu_count() or 1
        worker_candidates = _binary_tree_worker_candidates(hardware_workers)
        tile_candidates = _candidate_tile_sizes(workload.cols)
        trials: list[TrialRecord] = []
        best_trial: TrialRecord | None = None
        deadline = time.monotonic() + max(time_budget_seconds, 1.0)

        for worker_count in worker_candidates:
            if time.monotonic() >= deadline:
                notes.append("time budget reached before finishing the worker sweep")
                break

            # Every worker candidate gets a full tile-size sweep, as requested.
            for tile_size in tile_candidates:
                if time.monotonic() >= deadline:
                    notes.append(f"time budget reached while sweeping tile sizes for workers={worker_count}")
                    break

                trial = self._measure_trial(
                    executable_path,
                    workload,
                    dataset,
                    reference_output,
                    requested_workers=worker_count,
                    tile_size=tile_size,
                    repeats=_default_repeats(workload),
                )
                trials.append(trial)
                if self._is_better(trial, best_trial):
                    best_trial = trial

        selected_config = None if best_trial is None else dict(best_trial.config)
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=selected_config,
            best_trial=best_trial,
            trials=trials,
            notes=notes,
        )

    def _measure_trial(
        self,
        executable_path: Path,
        workload: WorkloadSpec,
        dataset: DatasetPaths,
        reference_output: list[float],
        *,
        requested_workers: int,
        tile_size: int,
        repeats: int,
    ) -> TrialRecord:
        command = [
            str(executable_path),
            "--matrix",
            str(dataset.matrix_path),
            "--vector",
            str(dataset.vector_path),
            "--output",
            str(CPU_OUTPUT_PATH),
            "--rows",
            str(workload.rows),
            "--cols",
            str(workload.cols),
            "--workers",
            str(requested_workers),
            "--tile-size",
            str(tile_size),
            "--repeats",
            str(repeats),
        ]

        notes: list[str] = []
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
        except subprocess.TimeoutExpired:
            return TrialRecord(
                backend=self.name,
                config={"requested_workers": requested_workers, "tile_size": tile_size},
                elapsed_seconds=float("inf"),
                throughput_gflops=0.0,
                score=0.0,
                verified=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                notes=["CPU trial timed out"],
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            return TrialRecord(
                backend=self.name,
                config={"requested_workers": requested_workers, "tile_size": tile_size},
                elapsed_seconds=float("inf"),
                throughput_gflops=0.0,
                score=0.0,
                verified=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                notes=[stderr if stderr else str(exc)],
            )

        metrics = json.loads(completed.stdout)
        candidate_output = load_float32_file(CPU_OUTPUT_PATH)
        verified, max_abs_error, max_rel_error = compare_vectors(
            reference_output,
            candidate_output,
            atol=workload.verify_atol,
            rtol=workload.verify_rtol,
        )

        elapsed_seconds = float(metrics["per_run_seconds"])
        score = linear_time_score(
            elapsed_seconds,
            ideal_seconds=workload.ideal_seconds,
            zero_score_seconds=workload.zero_score_seconds,
        )
        if not verified:
            score = 0.0
            notes.append("verification failed")

        actual_workers = int(metrics["actual_workers"])
        throughput_gflops = workload.flops_per_run / max(elapsed_seconds, 1e-12) / 1.0e9
        config = {
            "workers": actual_workers,
            "requested_workers": int(metrics["requested_workers"]),
            "tile_size": int(metrics["tile_size"]),
            "repeats": int(metrics["repeats"]),
        }
        return TrialRecord(
            backend=self.name,
            config=config,
            elapsed_seconds=elapsed_seconds,
            throughput_gflops=throughput_gflops,
            score=score,
            verified=verified,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
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
                    "call "
                    f"\"{vsdevcmd_path}\" -arch=x64 -host_arch=x64 >nul",
                    "cl /nologo /std:c++20 /O2 /EHsc "
                    f"/Fe:\"{CPU_EXECUTABLE_PATH}\" "
                    f"\"{CPU_SOURCE_PATH}\"",
                ]
            )
            + "\n",
            encoding="ascii",
        )

        # PowerShell's call operator handles a batch-file path with spaces more
        # reliably than invoking cmd.exe directly through subprocess argument
        # quoting, which is important in this workspace because
        # "performance metrics" is part of the path.
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

    @staticmethod
    def _is_better(candidate: TrialRecord | None, incumbent: TrialRecord | None) -> bool:
        if candidate is None:
            return False
        if incumbent is None:
            return candidate.verified
        if candidate.verified != incumbent.verified:
            return candidate.verified
        if candidate.score != incumbent.score:
            return candidate.score > incumbent.score
        return candidate.elapsed_seconds < incumbent.elapsed_seconds
