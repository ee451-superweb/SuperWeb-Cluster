"""Runtime task execution for compute-node processors."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from common.work_partition import partition_contiguous_range
from compute_node.input_matrix import build_dataset_layout, build_input_matrix_spec, dataset_is_generated
from compute_node.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile, load_runtime_processor_inventory
from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
from wire.runtime import TaskAssign, TaskResult

ROOT_DIR = Path(__file__).resolve().parent
INPUT_MATRIX_GENERATED_DIR = ROOT_DIR / "input_matrix" / "generated"
PERFORMANCE_METRICS_DIR = ROOT_DIR / "performance_metrics"
CPU_WINDOWS_EXECUTABLE = (
    PERFORMANCE_METRICS_DIR
    / "fixed_matrix_vector_multiplication"
    / "cpu"
    / "windows"
    / "build"
    / "fmvm_cpu_windows.exe"
)
CPU_MACOS_EXECUTABLE = (
    PERFORMANCE_METRICS_DIR
    / "fixed_matrix_vector_multiplication"
    / "cpu"
    / "macos"
    / "build"
    / "fmvm_cpu_macos"
)
CUDA_EXECUTABLE = (
    PERFORMANCE_METRICS_DIR
    / "fixed_matrix_vector_multiplication"
    / "cuda"
    / "build"
    / ("fmvm_cuda_runner.exe" if sys.platform == "win32" else "fmvm_cuda_runner")
)


@dataclass(frozen=True, slots=True)
class ProcessorTaskSlice:
    """One local processor plus the rows it should compute."""

    processor: RuntimeProcessorProfile
    row_start: int
    row_end: int


class FixedMatrixVectorTaskExecutor:
    """Execute one fixed-matrix-vector task on the local compute node."""

    def __init__(
        self,
        inventory: RuntimeProcessorInventory | None = None,
        *,
        dataset_root: Path | None = None,
    ) -> None:
        self.inventory = inventory or load_runtime_processor_inventory()
        self.spec = build_input_matrix_spec()
        self.dataset_root = INPUT_MATRIX_GENERATED_DIR if dataset_root is None else Path(dataset_root)
        self.dataset_layout = build_dataset_layout(self.dataset_root)

    def execute_task(self, task: TaskAssign) -> TaskResult:
        """Run one assigned row slice and return its output bytes."""

        self._validate_task(task)
        processor_slices = self._build_local_processor_slices(task)

        with tempfile.TemporaryDirectory(prefix="superweb-task-") as temp_dir:
            temp_root = Path(temp_dir)
            vector_path = temp_root / "x.bin"
            vector_path.write_bytes(task.vector_data)

            with ThreadPoolExecutor(max_workers=len(processor_slices), thread_name_prefix="local-processor") as executor:
                output_chunks = list(
                    executor.map(
                        lambda processor_slice: (
                            processor_slice.row_start,
                            self._run_processor_slice(processor_slice, task.iteration_count, vector_path, temp_root),
                        ),
                        processor_slices,
                    )
                )

        merged = b"".join(chunk for _, chunk in sorted(output_chunks, key=lambda item: item[0]))
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=200,
            row_start=task.row_start,
            row_end=task.row_end,
            output_length=task.row_end - task.row_start,
            output_vector=merged,
            iteration_count=task.iteration_count,
        )

    def _validate_task(self, task: TaskAssign) -> None:
        if task.method != METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
            raise ValueError(f"unsupported task method: {task.method}")
        if task.row_start < 0 or task.row_end > self.spec.rows or task.row_end <= task.row_start:
            raise ValueError("task row range is invalid")
        if task.vector_length != self.spec.cols:
            raise ValueError(f"task vector length {task.vector_length} does not match expected {self.spec.cols}")
        if len(task.vector_data) != task.vector_length * 4:
            raise ValueError("task vector byte size does not match vector_length")
        if task.iteration_count <= 0:
            raise ValueError("task iteration_count must be positive")
        if not dataset_is_generated(self.dataset_layout, self.spec):
            raise FileNotFoundError(
                f"missing generated input matrix at {self.dataset_layout.root_dir}; run compute_node/input_matrix/generate.py"
            )
        if not self.inventory.processors:
            raise RuntimeError("no locally registered processors are available for task execution")

    def _build_local_processor_slices(self, task: TaskAssign) -> list[ProcessorTaskSlice]:
        partitions = partition_contiguous_range(
            task.row_start,
            task.row_end,
            [processor.effective_gflops for processor in self.inventory.processors],
        )
        slices: list[ProcessorTaskSlice] = []
        for partition, processor in zip(partitions, self.inventory.processors):
            if partition.end <= partition.start:
                continue
            slices.append(
                ProcessorTaskSlice(
                    processor=processor,
                    row_start=partition.start,
                    row_end=partition.end,
                )
            )
        return slices

    def _run_processor_slice(
        self,
        processor_slice: ProcessorTaskSlice,
        iteration_count: int,
        vector_path: Path,
        temp_root: Path,
    ) -> bytes:
        output_path = temp_root / (
            f"{processor_slice.processor.hardware_type}_{processor_slice.row_start}_{processor_slice.row_end}.bin"
        )
        command = self._build_runtime_command(processor_slice, iteration_count, vector_path, output_path)
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PERFORMANCE_METRICS_DIR,
            timeout=300.0,
        )
        if not output_path.exists():
            stdout = (completed.stdout or "").strip()
            raise RuntimeError(
                f"{processor_slice.processor.hardware_type} runtime executable completed without writing {output_path.name}: "
                f"{stdout}"
            )
        raw = output_path.read_bytes()
        expected_bytes = (processor_slice.row_end - processor_slice.row_start) * 4
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{processor_slice.processor.hardware_type} runtime output has {len(raw)} bytes, expected {expected_bytes}"
            )
        return raw

    def _build_runtime_command(
        self,
        processor_slice: ProcessorTaskSlice,
        iteration_count: int,
        vector_path: Path,
        output_path: Path,
    ) -> list[str]:
        processor = processor_slice.processor
        if processor.hardware_type == "cpu":
            executable_path = CPU_WINDOWS_EXECUTABLE if sys.platform == "win32" else CPU_MACOS_EXECUTABLE
            workers = int(processor.best_config.get("workers") or processor.best_config.get("requested_workers") or 1)
            tile_size = int(processor.best_config.get("tile_size") or self.spec.cols)
            return [
                str(executable_path),
                "--matrix",
                str(self.dataset_layout.matrix_path),
                "--vector",
                str(vector_path),
                "--output",
                str(output_path),
                "--rows",
                str(self.spec.rows),
                "--cols",
                str(self.spec.cols),
                "--row-start",
                str(processor_slice.row_start),
                "--row-end",
                str(processor_slice.row_end),
                "--fixed-workers",
                str(workers),
                "--fixed-tile-size",
                str(tile_size),
                # Task mode uses iteration_count to repeat the same math locally
                # without resending the client request through the cluster.
                "--iteration-count",
                str(iteration_count),
            ]

        if processor.hardware_type == "cuda":
            executable_path = CUDA_EXECUTABLE
            transpose = 1 if bool(processor.best_config.get("transpose")) else 0
            block_size = int(processor.best_config.get("block_size") or 256)
            tile_size = int(processor.best_config.get("tile_size") or 1)
            return [
                str(executable_path),
                "--matrix",
                str(self.dataset_layout.matrix_path),
                "--vector",
                str(vector_path),
                "--output",
                str(output_path),
                "--rows",
                str(self.spec.rows),
                "--cols",
                str(self.spec.cols),
                "--row-start",
                str(processor_slice.row_start),
                "--row-end",
                str(processor_slice.row_end),
                "--fixed-transpose",
                str(transpose),
                "--fixed-block-size",
                str(block_size),
                "--fixed-tile-size",
                str(tile_size),
                # Task mode uses iteration_count to repeat the same math locally
                # without resending the client request through the cluster.
                "--iteration-count",
                str(iteration_count),
            ]

        raise ValueError(f"unsupported local processor type: {processor.hardware_type}")


