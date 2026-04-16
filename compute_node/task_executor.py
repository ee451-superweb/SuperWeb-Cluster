"""Runtime task execution for compute-node processors."""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.constants import DX12_BACKEND_DISABLED_REASON
from common.work_partition import partition_contiguous_range
from compute_node.compute_methods.fixed_matrix_vector_multiplication import (
    CPU_MACOS_EXECUTABLE_PATH,
    CPU_WINDOWS_EXECUTABLE_PATH,
    CUDA_EXECUTABLE_PATH,
    FMVM_METHOD_DIR,
)
from compute_node.input_matrix.fixed_matrix_vector_multiplication import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.config import DATASET_DIR as FMVM_DATASET_DIR
from compute_node.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile, load_runtime_processor_inventory
from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
from wire.runtime import TaskAssign, TaskResult

ROOT_DIR = Path(__file__).resolve().parent
INPUT_MATRIX_GENERATED_DIR = FMVM_DATASET_DIR


@dataclass(frozen=True, slots=True)
class ProcessorTaskSlice:
    """One local processor plus the rows it should compute."""

    processor: RuntimeProcessorProfile
    row_start: int
    row_end: int


class _Dx12ResidentRunner:
    """Keep the DX12 matrix resident for the full compute-node process lifetime."""

    def __init__(self, dataset_layout, spec, processors: tuple[RuntimeProcessorProfile, ...]) -> None:
        thread_group_sizes = sorted(
            {
                int(processor.best_config.get("thread_group_size") or 256)
                for processor in processors
                if processor.hardware_type == "dx12"
            }
        )
        if not thread_group_sizes:
            raise ValueError("DX12 resident runner requires at least one dx12 processor profile")

        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                str(DX12_EXECUTABLE_PATH),
                "--server",
                "1",
                "--matrix",
                str(dataset_layout.matrix_path),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--thread-group-sizes",
                ",".join(str(value) for value in thread_group_sizes),
            ],
            cwd=FMVM_METHOD_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready_line = self._read_response_line()
        if not ready_line.startswith("READY\t"):
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner failed to start: {ready_line or stderr or 'unknown error'}")
        atexit.register(self.close)

    def run_slice(
        self,
        *,
        vector_path: Path,
        output_path: Path,
        row_start: int,
        row_end: int,
        thread_group_size: int,
        rows_per_thread: int,
        iteration_count: int,
    ) -> dict[str, object]:
        with self._lock:
            self._ensure_running()
            command = "\t".join(
                [
                    "RUN",
                    str(vector_path),
                    str(output_path),
                    str(row_start),
                    str(row_end),
                    str(thread_group_size),
                    str(rows_per_thread),
                    str(iteration_count),
                ]
            )
            assert self._process.stdin is not None
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            response_line = self._read_response_line()
            if response_line.startswith("OK\t"):
                return json.loads(response_line.split("\t", 1)[1])
            if response_line.startswith("ERR\t"):
                raise RuntimeError(response_line.split("\t", 1)[1])
            raise RuntimeError(f"unexpected DX12 resident runner response: {response_line}")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None

    def _ensure_running(self) -> None:
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner exited unexpectedly: {stderr or self._process.returncode}")

    def _read_response_line(self) -> str:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner closed its pipe unexpectedly: {stderr or 'no output'}")
        return line.rstrip("\r\n")


class FixedMatrixVectorTaskExecutor:
    """Execute one fixed-matrix-vector task on the local compute node."""

    def __init__(
        self,
        inventory: RuntimeProcessorInventory | None = None,
        *,
        dataset_root: Path | None = None,
    ) -> None:
        self.inventory = inventory or load_runtime_processor_inventory()
        self.spec = build_input_matrix_spec(default_variant="runtime")
        self.dataset_root = INPUT_MATRIX_GENERATED_DIR if dataset_root is None else Path(dataset_root)
        self.dataset_layout = build_dataset_layout(self.dataset_root, prefix="runtime_")
        self._dx12_runner = self._build_dx12_resident_runner()

    def close(self) -> None:
        if self._dx12_runner is not None:
            self._dx12_runner.close()
            self._dx12_runner = None

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
        if processor_slice.processor.hardware_type == "dx12" and self._dx12_runner is not None:
            self._dx12_runner.run_slice(
                vector_path=vector_path,
                output_path=output_path,
                row_start=processor_slice.row_start,
                row_end=processor_slice.row_end,
                thread_group_size=int(processor_slice.processor.best_config.get("thread_group_size") or 256),
                rows_per_thread=int(processor_slice.processor.best_config.get("rows_per_thread") or 1),
                iteration_count=iteration_count,
            )
        else:
            command = self._build_runtime_command(processor_slice, iteration_count, vector_path, output_path)
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=FMVM_METHOD_DIR,
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

    def _build_dx12_resident_runner(self) -> _Dx12ResidentRunner | None:
        dx12_processors = tuple(
            processor for processor in self.inventory.processors if processor.hardware_type == "dx12"
        )
        if not dx12_processors or sys.platform != "win32":
            return None
        return None

    def _build_runtime_command(
        self,
        processor_slice: ProcessorTaskSlice,
        iteration_count: int,
        vector_path: Path,
        output_path: Path,
    ) -> list[str]:
        processor = processor_slice.processor
        if processor.hardware_type == "cpu":
            executable_path = CPU_WINDOWS_EXECUTABLE_PATH if sys.platform == "win32" else CPU_MACOS_EXECUTABLE_PATH
            workers = int(processor.best_config.get("workers") or processor.best_config.get("requested_workers") or 1)
            tile_size = int(processor.best_config.get("tile_size") or self.spec.cols)
            accumulation_precision = str(processor.best_config.get("accumulation_precision") or "fp32")
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
                "--accumulation-precision",
                accumulation_precision,
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
            executable_path = CUDA_EXECUTABLE_PATH
            transpose = 1 if bool(processor.best_config.get("transpose")) else 0
            block_size = int(processor.best_config.get("block_size") or 256)
            tile_size = int(processor.best_config.get("tile_size") or 1)
            accumulation_precision = str(processor.best_config.get("accumulation_precision") or "fp32")
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
                "--accumulation-precision",
                accumulation_precision,
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

        if processor.hardware_type == "dx12":
            raise ValueError(DX12_BACKEND_DISABLED_REASON)

        raise ValueError(f"unsupported local processor type: {processor.hardware_type}")


