"""Execute spatial-convolution runtime tasks on the local compute node.

Use this module when a compute node receives a spatial-convolution
``TaskAssign`` and needs to choose the best local backend, run the native
runner, and package the result back into a ``TaskResult``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.constants import (
    DEFAULT_SPATIAL_CUDA_COOLDOWN_MS,
    DX12_BACKEND_DISABLED_REASON,
    METHOD_SPATIAL_CONVOLUTION,
    STATUS_OK,
)
from app.compute_resource_policy import resolve_capped_cpu_worker_count
from compute_node.compute_methods.spatial_convolution.paths import (
    CPU_MACOS_EXECUTABLE_PATH,
    CPU_WINDOWS_EXECUTABLE_PATH,
    CUDA_EXECUTABLE_PATH,
    METAL_EXECUTABLE_PATH,
    SPATIAL_METHOD_DIR,
)
from compute_node.performance_metrics.spatial_convolution.models import BenchmarkSpec
from compute_node.performance_metrics.spatial_convolution.config import (
    DATASET_DIR as TOP_LEVEL_DATASET_DIR,
    GENERATE_SCRIPT_PATH as TOP_LEVEL_GENERATE_SCRIPT_PATH,
    RESULT_PATH as TOP_LEVEL_RESULT_PATH,
)
from compute_node.performance_metrics.spatial_convolution.workloads import get_runtime_spec, get_test_spec
from compute_node.performance_summary import RuntimeProcessorProfile, load_runtime_processor_inventory
from setup import active_python_path
from wire.internal_protocol.runtime_transport import TaskAssign, TaskResult, TransferMode

METHOD_DIR = SPATIAL_METHOD_DIR
DEFAULT_CONV_RESULT_PATH = TOP_LEVEL_RESULT_PATH
DEFAULT_DATASET_DIR = TOP_LEVEL_DATASET_DIR
DATASET_GENERATE_SCRIPT_PATH = TOP_LEVEL_GENERATE_SCRIPT_PATH
DISABLED_SPATIAL_BACKENDS = frozenset({"dx12"})
SPATIAL_CUDA_COOLDOWN_MS = DEFAULT_SPATIAL_CUDA_COOLDOWN_MS


def _best_backend_profile(result_path: Path) -> RuntimeProcessorProfile | None:
    """Return the best usable processor profile from a method result file.

    Args:
        result_path: Spatial benchmark result path for this compute node.

    Returns:
        The best non-disabled processor profile, or ``None``.
    """
    try:
        inventory = load_runtime_processor_inventory(
            result_path=result_path,
            method=METHOD_SPATIAL_CONVOLUTION,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    for processor in inventory.processors:
        if processor.hardware_type not in DISABLED_SPATIAL_BACKENDS:
            return processor
    return None


def _best_backend_name(result_path: Path) -> str:
    """Return the best usable backend name from the method result file.

    Args:
        result_path: Spatial benchmark result path for this compute node.

    Returns:
        The preferred backend name, falling back to ``cpu`` when unknown.
    """
    profile = _best_backend_profile(result_path)
    if profile is not None:
        return profile.hardware_type
    if not result_path.exists():
        return "cpu"

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "cpu"

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
    """Resolve the native runner executable path for one backend name.

    Args:
        backend_name: Backend name such as ``cpu`` or ``cuda``.

    Returns:
        The executable path for the requested backend.
    """
    if backend_name == "cpu":
        if sys.platform == "win32":
            return CPU_WINDOWS_EXECUTABLE_PATH
        if sys.platform == "darwin":
            return CPU_MACOS_EXECUTABLE_PATH
    if backend_name == "cuda":
        return CUDA_EXECUTABLE_PATH
    if backend_name == "dx12":
        raise ValueError(DX12_BACKEND_DISABLED_REASON)
    if backend_name == "metal":
        return METAL_EXECUTABLE_PATH
    raise ValueError(f"unsupported spatial_convolution backend: {backend_name}")


def _spec_for_task(task: TaskAssign):
    """Resolve the workload spec and dataset variant for one task assignment."""
    return load_named_workload_spec(task.object_id)


def load_named_workload_spec(object_id: str):
    """Resolve one supported spatial-convolution object id into its workload spec.

    Args:
        object_id: Object id carried by the task assignment.

    Returns:
        A ``(BenchmarkSpec, variant_name)`` pair for the requested object id.
    """

    if object_id.endswith("/runtime"):
        return get_runtime_spec(), "runtime"
    if object_id.endswith("/test"):
        return get_test_spec(), "test"
    raise ValueError(
        "spatial_convolution requests currently require object_id ending with '/test' or '/runtime'"
    )


def _validate_task_against_spec(task: TaskAssign, spec) -> None:
    """Validate that a spatial task matches the resolved workload spec.

    Args:
        task: Spatial task assignment received from the main node.
        spec: Benchmark spec resolved from the task object id.

    Returns:
        ``None`` when the task is valid for local execution.
    """
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
    """Generate the shared spatial dataset if the compute node is missing it.

    Args:
        dataset_dir: Directory that should hold spatial benchmark datasets.

    Returns:
        ``None`` after the required input files exist.
    """
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
        """Store the benchmark result path and dataset root used at runtime.

        Args:
            result_path: Optional spatial benchmark result path override.
            dataset_root: Optional spatial dataset directory override.
        """
        self.result_path = DEFAULT_CONV_RESULT_PATH if result_path is None else Path(result_path)
        self.dataset_root = DEFAULT_DATASET_DIR if dataset_root is None else Path(dataset_root)

    def execute_task(self, task: TaskAssign) -> TaskResult:
        """Execute one spatial-convolution task and return its result payload.

        Args:
            task: Spatial task assignment received from the main node.

        Returns:
            A ``TaskResult`` containing the output-channel slice bytes.
        """
        spec, variant = _spec_for_task(task)
        _validate_task_against_spec(task, spec)
        _ensure_dataset_ready(self.dataset_root)
        backend_profile = _best_backend_profile(self.result_path)
        backend_name = backend_profile.hardware_type if backend_profile is not None else _best_backend_name(self.result_path)
        best_config = {} if backend_profile is None else dict(backend_profile.best_config)
        executable_path = _runner_path(backend_name)
        if not executable_path.exists():
            raise FileNotFoundError(f"spatial_convolution runner is missing: {executable_path}")

        input_path = self.dataset_root / f"{variant}_input.bin"
        if not input_path.exists():
            raise FileNotFoundError(f"missing generated conv2d input at {input_path}")

        disk_first_result = task.transfer_mode in (
            TransferMode.ARTIFACT_PREFERRED,
            TransferMode.ARTIFACT_REQUIRED,
        )

        with tempfile.TemporaryDirectory(prefix="superweb-spatial-task-") as temp_dir:
            temp_root = Path(temp_dir)
            weight_path = temp_root / "weight_slice.bin"
            weight_path.write_bytes(task.weight_data)
            output_path = temp_root / "output_slice.bin"
            if disk_first_result:
                output_fd, output_name = tempfile.mkstemp(prefix="superweb-spatial-output-", suffix=".bin")
                os.close(output_fd)
                output_path = Path(output_name)

            try:
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
                    configured_workers = int(
                        best_config.get("workers")
                        or best_config.get("requested_workers")
                        or resolve_capped_cpu_worker_count()
                    )
                    worker_count = max(1, min(resolve_capped_cpu_worker_count(), configured_workers))
                    cmd.extend(
                        [
                            "--start-oc",
                            str(task.start_oc),
                            "--end-oc",
                            str(task.end_oc),
                            "--workers",
                            str(worker_count),
                        ]
                    )
                elif backend_name == "cuda":
                    configured_batch = int(best_config.get("output_channel_batch") or 0)
                    resolved_batch = max(
                        1,
                        min(task.end_oc - task.start_oc, configured_batch or (task.end_oc - task.start_oc)),
                    )
                    cmd.extend(
                        [
                            "--output-channel-batch",
                            str(resolved_batch),
                            "--cooldown-ms",
                            str(SPATIAL_CUDA_COOLDOWN_MS),
                        ]
                    )

                completed = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=METHOD_DIR,
                    timeout=900.0,
                )
                if not output_path.exists():
                    raise RuntimeError(
                        f"{backend_name} spatial_convolution runner completed without writing {output_path.name}: "
                        f"{(completed.stdout or '').strip()}"
                    )

                output_size = output_path.stat().st_size
                conv_spec = BenchmarkSpec(
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
                if output_size != expected_output_bytes:
                    raise ValueError(
                        f"spatial_convolution runtime output has {output_size} bytes, expected {expected_output_bytes}"
                    )

                output_bytes = b"" if disk_first_result else output_path.read_bytes()
                local_result_path = str(output_path) if disk_first_result else ""
            except Exception:
                if disk_first_result:
                    output_path.unlink(missing_ok=True)
                raise

        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            local_result_path=local_result_path,
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
