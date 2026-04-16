"""Metal backend for the Convolution benchmark on macOS."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from compute_node.compute_methods.spatial_convolution.performance_metrics.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.path_utils import (
    sanitize_text,
    to_relative_cli_path,
    to_relative_string,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.scoring import (
    linear_time_score,
)
from compute_node.performance_metrics.benchmark_status import emit_status

ROOT_DIR = Path(__file__).resolve().parents[1]
METAL_DIR = ROOT_DIR / "conv2d_runners" / "metal"
METAL_HOST_SOURCE_PATH = METAL_DIR / "fmvm_metal_runner.mm"
METAL_KERNEL_SOURCE_PATH = METAL_DIR / "fmvm_metal_kernels.metal"
METAL_BUILD_DIR = METAL_DIR / "build"
METAL_EXECUTABLE_PATH = METAL_BUILD_DIR / "fmvm_metal_runner"


def _relative_project_path(path: Path) -> str:
    return to_relative_string(path, start=ROOT_DIR)

def _relative_cli_path(path: Path) -> str:
    return to_relative_cli_path(path, start=ROOT_DIR)

def _candidate_block_sizes() -> list[int]:
    return [32, 64, 128, 256, 512, 1024]

def _candidate_tile_sizes() -> list[int]:
    return [1, 2, 4, 8, 16]

class MetalBackend:
    name = "metal"

    def diagnostic_context(self, _spec: BenchmarkSpec | None = None) -> dict[str, object]:
        return {
            "block_size_candidates": _candidate_block_sizes(),
            "tile_size_candidates": _candidate_tile_sizes(),
            "autotune_repeats": DEFAULT_AUTOTUNE_REPEATS,
            "measurement_repeats": DEFAULT_MEASUREMENT_REPEATS,
            "runner_path": str(METAL_EXECUTABLE_PATH),
        }

    def probe(self) -> tuple[bool, str]:
        if sys.platform != "darwin": return False, "Metal backend is only available on macOS."
        return True, "Metal backend ready"

    def run(
        self,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        measurement_spec: BenchmarkSpec | None = None,
        measurement_dataset: DatasetLayout | None = None,
        time_budget_seconds: float,
        force_rebuild: bool = False,
    ) -> BackendResult:
        available, message = self.probe()
        notes = [message]
        if not available: return BackendResult(self.name, False, None, None, None, [], notes)

        measurement_spec = spec if measurement_spec is None else measurement_spec
        measurement_dataset = dataset if measurement_dataset is None else measurement_dataset
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else DEFAULT_MEASUREMENT_REPEATS
        )
        executable_path = METAL_EXECUTABLE_PATH # Skip dynamic build for brevity

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                block_sizes=_candidate_block_sizes(),
                tile_sizes=_candidate_tile_sizes(),
                autotune_repeats=DEFAULT_AUTOTUNE_REPEATS,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
            measurement_metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                block_sizes=[int(autotune_metrics["block_size"])],
                tile_sizes=[int(autotune_metrics["tile_size"])],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except Exception as exc:
            notes.append("Metal benchmark failed or timed out")
            return BackendResult(self.name, False, None, None, None, [], notes)

        if measurement_spec != spec or measurement_dataset != dataset:
            notes.append(f"Autotuned on {spec.name} and measured on {measurement_spec.name}.")

        autotune_score = linear_time_score(
            float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds
        )
        measurement_score = linear_time_score(
            float(measurement_metrics["measurement_wall_clock_latency_seconds"]),
            ideal_seconds=measurement_spec.ideal_seconds,
            zero_score_seconds=measurement_spec.zero_score_seconds
        )

        autotune_config = {
            "block_size": int(autotune_metrics["block_size"]), "tile_size": int(autotune_metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]), "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        measurement_config = {
            "block_size": int(measurement_metrics["block_size"]), "tile_size": int(measurement_metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]), "measurement_repeats": int(measurement_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }

        autotune_trial = TrialRecord(self.name, autotune_config, float(autotune_metrics["autotune_wall_clock_latency_seconds"]), float(autotune_metrics["autotune_effective_gflops"]), str(autotune_metrics["autotune_checksum"]), autotune_score, [])
        trial = TrialRecord(self.name, measurement_config, float(measurement_metrics["measurement_wall_clock_latency_seconds"]), float(measurement_metrics["measurement_effective_gflops"]), str(measurement_metrics["measurement_checksum"]), measurement_score, [])
        return BackendResult(self.name, True, dict(measurement_config), autotune_trial, trial, [autotune_trial, trial], notes)

    def _run_runner(
        self,
        executable_path: Path,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        block_sizes: list[int],
        tile_sizes: list[int],
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        command = [
            str(executable_path),
            "--input", _relative_cli_path(dataset.input_path),
            "--weight", _relative_cli_path(dataset.weight_path),
            "--h", str(spec.h), "--w", str(spec.w),
            "--cin", str(spec.c_in), "--cout", str(spec.c_out),
            "--k", str(spec.k), "--pad", str(spec.pad),
            "--stride", str(spec.stride),
            "--block-sizes", ",".join(str(v) for v in block_sizes),
            "--tile-sizes", ",".join(str(v) for v in tile_sizes),
            "--autotune-repeats", str(autotune_repeats),
            "--measurement-repeats", str(measurement_repeats),
        ]
        emit_status(
            "method.spatial_convolution.backend.native_runner.start",
            status="running",
            method="spatial_convolution",
            backend=self.name,
            command=command,
            timeout_seconds=timeout_seconds,
            autotune_repeats=autotune_repeats,
            measurement_repeats=measurement_repeats,
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=ROOT_DIR,
        )
        emit_status(
            "method.spatial_convolution.backend.native_runner.complete",
            status="running",
            method="spatial_convolution",
            backend=self.name,
            command=command,
        )
        return json.loads(completed.stdout)
