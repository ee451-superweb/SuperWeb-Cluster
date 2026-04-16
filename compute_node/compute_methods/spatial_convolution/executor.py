"""Runtime task execution for the spatial-convolution compute method."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.constants import DX12_BACKEND_DISABLED_REASON, METHOD_SPATIAL_CONVOLUTION, STATUS_OK
from setup import active_python_path
from compute_node.performance_metrics.spatial_convolution.config import (
    DATASET_DIR as TOP_LEVEL_DATASET_DIR,
    GENERATE_SCRIPT_PATH as TOP_LEVEL_GENERATE_SCRIPT_PATH,
    RESULT_PATH as TOP_LEVEL_RESULT_PATH,
)
from wire.runtime import TaskAssign, TaskResult

METHOD_DIR = Path(__file__).resolve().parent
PERF_DIR = METHOD_DIR / "performance_metrics"
DEFAULT_CONV_RESULT_PATH = TOP_LEVEL_RESULT_PATH
DEFAULT_DATASET_DIR = TOP_LEVEL_DATASET_DIR
DATASET_GENERATE_SCRIPT_PATH = TOP_LEVEL_GENERATE_SCRIPT_PATH
DISABLED_SPATIAL_BACKENDS = frozenset({"dx12"})


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_workloads_module():
    return _load_module("superweb_cluster_spatial_conv_workloads", PERF_DIR / "workloads.py")


def _load_models_module():
    return _load_module("superweb_cluster_spatial_conv_models", PERF_DIR / "models.py")


def _best_backend_name(result_path: Path) -> str:
    if not result_path.exists():
        return "cpu"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if isinstance(methods, dict):
        method_payload = methods.get(METHOD_SPATIAL_CONVOLUTION) or {}
        ranking = method_payload.get("ranking", [])
        for backend_name in ranking:
            name = str(backend_name)
            if name not in DISABLED_SPATIAL_BACKENDS:
                return name
        return "cpu"
    ranking = payload.get("ranking", [])
    for backend_name in ranking:
        name = str(backend_name)
        if name not in DISABLED_SPATIAL_BACKENDS:
            return name
    return "cpu"


def _runner_path(backend_name: str) -> Path:
    runners_dir = PERF_DIR / "conv2d_runners"
    if backend_name == "cpu":
        if sys.platform == "win32":
            return runners_dir / "cpu" / "windows" / "build" / "fmvm_cpu_windows.exe"
        if sys.platform == "darwin":
            return runners_dir / "cpu" / "macos" / "build" / "fmvm_cpu_macos"
    if backend_name == "cuda":
        return runners_dir / "cuda" / "build" / ("fmvm_cuda_runner.exe" if os.name == "nt" else "fmvm_cuda_runner")
    if backend_name == "dx12":
        raise ValueError(DX12_BACKEND_DISABLED_REASON)
    if backend_name == "metal":
        return runners_dir / "metal" / "build" / "fmvm_metal_runner"
    raise ValueError(f"unsupported spatial_convolution backend: {backend_name}")


def _spec_for_task(task: TaskAssign):
    return load_named_workload_spec(task.object_id)


def load_named_workload_spec(object_id: str):
    """Resolve one supported spatial-convolution object id into its workload spec."""

    workloads = _load_workloads_module()
    if object_id.endswith("/runtime"):
        return workloads.get_runtime_spec(), "runtime"
    if object_id.endswith("/test"):
        return workloads.get_test_spec(), "test"
    raise ValueError(
        "spatial_convolution requests currently require object_id ending with '/test' or '/runtime'"
    )


def _validate_task_against_spec(task: TaskAssign, spec) -> None:
    if task.method != METHOD_SPATIAL_CONVOLUTION:
        raise ValueError(f"unsupported task method: {task.method}")
    if task.start_oc < 0 or task.end_oc > spec.c_out or task.end_oc <= task.start_oc:
        raise ValueError("task output-channel range is invalid")
    if task.tensor_h and task.tensor_h != spec.h:
        raise ValueError(f"task tensor_h {task.tensor_h} does not match expected {spec.h}")
    if task.tensor_w and task.tensor_w != spec.w:
        raise ValueError(f"task tensor_w {task.tensor_w} does not match expected {spec.w}")
    if task.channels_in and task.channels_in != spec.c_in:
        raise ValueError(f"task channels_in {task.channels_in} does not match expected {spec.c_in}")
    if task.channels_out and task.channels_out != spec.c_out:
        raise ValueError(f"task channels_out {task.channels_out} does not match expected {spec.c_out}")
    if task.kernel_size and task.kernel_size != spec.k:
        raise ValueError(f"task kernel_size {task.kernel_size} does not match expected {spec.k}")
    if task.padding and task.padding != spec.pad:
        raise ValueError(f"task padding {task.padding} does not match expected {spec.pad}")
    if task.stride and task.stride != spec.stride:
        raise ValueError(f"task stride {task.stride} does not match expected {spec.stride}")

    expected_weight_bytes = spec.k * spec.k * spec.c_in * (task.end_oc - task.start_oc) * 4
    if len(task.weight_data) != expected_weight_bytes:
        raise ValueError(
            f"task weight byte size {len(task.weight_data)} does not match expected {expected_weight_bytes}"
        )


def _ensure_dataset_ready(dataset_dir: Path) -> None:
    if (dataset_dir / "runtime_input.bin").exists() and (dataset_dir / "test_input.bin").exists():
        return
    subprocess.run(
        [
            str(active_python_path()),
            str(DATASET_GENERATE_SCRIPT_PATH),
            "--output-dir",
            str(dataset_dir),
            "--role",
            "compute",
        ],
        check=True,
        cwd=METHOD_DIR,
        timeout=600.0,
    )


class SpatialConvolutionTaskExecutor:
    """Execute one spatial-convolution task on the local compute node."""

    def __init__(
        self,
        *,
        result_path: Path | None = None,
        dataset_root: Path | None = None,
    ) -> None:
        self.result_path = DEFAULT_CONV_RESULT_PATH if result_path is None else Path(result_path)
        self.dataset_root = DEFAULT_DATASET_DIR if dataset_root is None else Path(dataset_root)

    def execute_task(self, task: TaskAssign) -> TaskResult:
        spec, variant = _spec_for_task(task)
        _validate_task_against_spec(task, spec)
        _ensure_dataset_ready(self.dataset_root)
        backend_name = _best_backend_name(self.result_path)
        executable_path = _runner_path(backend_name)
        if not executable_path.exists():
            raise FileNotFoundError(f"spatial_convolution runner is missing: {executable_path}")

        input_path = self.dataset_root / f"{variant}_input.bin"
        if not input_path.exists():
            raise FileNotFoundError(f"missing generated conv2d input at {input_path}")

        with tempfile.TemporaryDirectory(prefix="superweb-spatial-task-") as temp_dir:
            temp_root = Path(temp_dir)
            weight_path = temp_root / "weight_slice.bin"
            output_path = temp_root / "output_slice.bin"
            weight_path.write_bytes(task.weight_data)

            cmd = [
                str(executable_path),
                "--input",
                str(input_path),
                "--weight",
                str(weight_path),
                "--output",
                str(output_path),
                "--h",
                str(spec.h),
                "--w",
                str(spec.w),
                "--cin",
                str(spec.c_in),
                "--cout",
                str(spec.c_out if backend_name == "cpu" else (task.end_oc - task.start_oc)),
                "--k",
                str(spec.k),
                "--pad",
                str(spec.pad),
                "--stride",
                str(spec.stride),
                "--autotune-repeats",
                "1",
                "--measurement-repeats",
                str(max(1, task.iteration_count)),
            ]
            if backend_name == "cpu":
                cmd.extend(
                    [
                        "--start-oc",
                        str(task.start_oc),
                        "--end-oc",
                        str(task.end_oc),
                        "--workers",
                        str(os.cpu_count() or 4),
                    ]
                )

            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=PERF_DIR,
                timeout=900.0,
            )
            if not output_path.exists():
                raise RuntimeError(
                    f"{backend_name} spatial_convolution runner completed without writing {output_path.name}: "
                    f"{(completed.stdout or '').strip()}"
                )
            output_bytes = output_path.read_bytes()

        models = _load_models_module()
        conv_spec = models.BenchmarkSpec(
            name="task",
            h=spec.h,
            w=spec.w,
            c_in=spec.c_in,
            c_out=task.end_oc - task.start_oc,
            k=spec.k,
            pad=spec.pad,
            ideal_seconds=1.0,
            zero_score_seconds=10.0,
            stride=spec.stride,
        )
        expected_output_bytes = conv_spec.output_bytes
        if len(output_bytes) != expected_output_bytes:
            raise ValueError(
                f"spatial_convolution runtime output has {len(output_bytes)} bytes, expected {expected_output_bytes}"
            )

        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            row_start=0,
            row_end=0,
            output_length=conv_spec.output_h * conv_spec.output_w * (task.end_oc - task.start_oc),
            output_vector=output_bytes,
            iteration_count=task.iteration_count,
            start_oc=task.start_oc,
            end_oc=task.end_oc,
            output_h=conv_spec.output_h,
            output_w=conv_spec.output_w,
        )
