"""Optional auto-tuned CUDA backend for fixed matrix-vector multiplication."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from fmvm_dataset import compare_vectors, ensure_transposed_matrix
from models import BackendResult, DatasetPaths, TrialRecord, WorkloadSpec
from scoring import linear_time_score

ROOT_DIR = Path(__file__).resolve().parents[1]
CUDA_SOURCE_PATH = (
    ROOT_DIR / "fixed_matrix_vector_multiplication" / "cuda" / "fmvm_cuda_runner.cu"
)
CUDA_BUILD_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "cuda" / "build"
CUDA_OUTPUT_PATH = CUDA_BUILD_DIR / "cuda_output.bin"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / (
    "fmvm_cuda_runner.exe" if os.name == "nt" else "fmvm_cuda_runner"
)


def _load_float32_vector(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(memoryview(raw).cast("f"))


def _candidate_block_sizes() -> list[int]:
    return [64, 128, 256, 512]


def _candidate_tile_sizes() -> list[int]:
    return [1, 2, 4, 8]


def _default_repeats(workload: WorkloadSpec) -> int:
    if workload.preset == "smoke":
        return 16
    if workload.preset == "quick":
        return 12
    if workload.preset == "standard":
        return 8
    return 4


class CudaBackend:
    """CUDA backend that compiles a small nvcc runner on demand."""

    name = "cuda"

    def probe(self) -> tuple[bool, str]:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None:
            return False, "nvcc was not found on PATH; CUDA backend skipped."
        if not CUDA_SOURCE_PATH.exists():
            return False, f"missing CUDA runner source at {CUDA_SOURCE_PATH}"
        return True, f"nvcc detected at {nvcc_path}."

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
            notes.append(f"failed to compile CUDA runner: {exc}")
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        default_config = {
            "transpose": False,
            "block_size": 256,
            "tile_size": 4,
            "repeats": _default_repeats(workload),
        }
        current_config = dict(default_config)
        trials: list[TrialRecord] = []
        best_trial: TrialRecord | None = None
        deadline = time.monotonic() + max(time_budget_seconds, 1.0)

        search_axes: list[tuple[str, list[object]]] = [
            ("transpose", [False, True]),
            ("block_size", _candidate_block_sizes()),
            ("tile_size", _candidate_tile_sizes()),
        ]

        for axis_name, axis_values in search_axes:
            axis_best = best_trial
            axis_best_config = dict(current_config)
            for axis_value in axis_values:
                if time.monotonic() >= deadline:
                    notes.append(f"time budget reached while tuning {axis_name}")
                    break

                candidate_config = dict(current_config)
                candidate_config[axis_name] = axis_value
                trial = self._measure_trial(
                    executable_path,
                    workload,
                    dataset,
                    reference_output,
                    candidate_config,
                    timeout_seconds=max(10.0, deadline - time.monotonic()),
                )
                trials.append(trial)
                if self._is_better(trial, axis_best):
                    axis_best = trial
                    axis_best_config = dict(candidate_config)

            current_config = axis_best_config
            if self._is_better(axis_best, best_trial):
                best_trial = axis_best

        selected_config = None if best_trial is None else dict(best_trial.config)
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=selected_config,
            best_trial=best_trial,
            trials=trials,
            notes=notes,
        )

    def _compile_if_needed(self) -> Path:
        CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        if CUDA_EXECUTABLE_PATH.exists() and CUDA_EXECUTABLE_PATH.stat().st_mtime >= CUDA_SOURCE_PATH.stat().st_mtime:
            return CUDA_EXECUTABLE_PATH

        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None:
            raise FileNotFoundError("nvcc not found")

        command = [
            nvcc_path,
            str(CUDA_SOURCE_PATH),
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-o",
            str(CUDA_EXECUTABLE_PATH),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return CUDA_EXECUTABLE_PATH

    def _measure_trial(
        self,
        executable_path: Path,
        workload: WorkloadSpec,
        dataset: DatasetPaths,
        reference_output: list[float],
        config: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> TrialRecord:
        matrix_path = dataset.matrix_path
        if bool(config["transpose"]):
            matrix_path = ensure_transposed_matrix(dataset, workload)

        CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable_path),
            "--matrix",
            str(matrix_path),
            "--vector",
            str(dataset.vector_path),
            "--output",
            str(CUDA_OUTPUT_PATH),
            "--rows",
            str(workload.rows),
            "--cols",
            str(workload.cols),
            "--transpose",
            "1" if bool(config["transpose"]) else "0",
            "--block-size",
            str(int(config["block_size"])),
            "--tile-size",
            str(int(config["tile_size"])),
            "--repeats",
            str(int(config["repeats"])),
        ]

        notes: list[str] = []
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(timeout_seconds, 1.0),
            )
        except subprocess.TimeoutExpired:
            return TrialRecord(
                backend=self.name,
                config=dict(config),
                elapsed_seconds=float("inf"),
                throughput_gflops=0.0,
                score=0.0,
                verified=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                notes=["CUDA trial timed out"],
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            note = stderr if stderr else str(exc)
            return TrialRecord(
                backend=self.name,
                config=dict(config),
                elapsed_seconds=float("inf"),
                throughput_gflops=0.0,
                score=0.0,
                verified=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                notes=[f"CUDA runner failed: {note}"],
            )

        metrics = json.loads(completed.stdout)
        elapsed_seconds = float(metrics["per_run_seconds"])
        candidate_output = _load_float32_vector(CUDA_OUTPUT_PATH)
        verified, max_abs_error, max_rel_error = compare_vectors(
            reference_output,
            candidate_output,
            atol=workload.verify_atol,
            rtol=workload.verify_rtol,
        )
        score = linear_time_score(
            elapsed_seconds,
            ideal_seconds=workload.ideal_seconds,
            zero_score_seconds=workload.zero_score_seconds,
        )
        if not verified:
            score = 0.0
            notes.append("verification failed")

        throughput_gflops = workload.flops_per_run / max(elapsed_seconds, 1e-12) / 1.0e9
        if "device_name" in metrics:
            notes.append(f"device={metrics['device_name']}")

        return TrialRecord(
            backend=self.name,
            config=dict(config),
            elapsed_seconds=elapsed_seconds,
            throughput_gflops=throughput_gflops,
            score=score,
            verified=verified,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            notes=notes,
        )

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
