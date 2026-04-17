"""Plan worker slices for one request from abstract per-method performance data.

Use this module when the main node has already accepted a client request and
needs to turn registered worker GFLOPS rankings into contiguous FMVM row ranges
or spatial-convolution output-channel ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from common.work_partition import partition_contiguous_range
from main_node.registry import RuntimePeerConnection, WorkerHardwareCapability


@dataclass(frozen=True, slots=True)
class WorkerTaskSlice:
    """Describe one worker assignment, including its logical task and artifact ids."""

    connection: RuntimePeerConnection
    task_id: str
    artifact_id: str
    row_start: int
    row_end: int
    effective_gflops: float
    method: str = METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION
    start_oc: int = 0
    end_oc: int = 0


class TaskDispatcher:
    """Convert registered worker performance into weighted request partitions."""

    def dispatch_fixed_matrix_vector_multiplication(
        self,
        *,
        request_id: str,
        rows: int,
        workers: list[RuntimePeerConnection],
        worker_hardware: list[WorkerHardwareCapability],
    ) -> list[WorkerTaskSlice]:
        """Use this when one FMVM request needs row partitions across active workers.

        Args: request_id logical task id, rows total matrix rows, workers live workers, worker_hardware FMVM performance entries.
        Returns: Weighted row slices, or an empty list when no worker has usable FMVM capacity.
        """
        worker_gflops: dict[str, float] = {worker.peer_id: 0.0 for worker in workers}
        for hardware in worker_hardware:
            worker_gflops[hardware.worker_peer_id] = worker_gflops.get(hardware.worker_peer_id, 0.0) + hardware.effective_gflops

        schedulable_workers = [worker for worker in workers if worker_gflops.get(worker.peer_id, 0.0) > 0.0]
        if not schedulable_workers:
            return []

        partitions = partition_contiguous_range(
            0,
            rows,
            [worker_gflops[worker.peer_id] for worker in schedulable_workers],
        )

        assignments: list[WorkerTaskSlice] = []
        for partition, worker in zip(partitions, schedulable_workers):
            if partition.end <= partition.start:
                continue
            assignments.append(
                WorkerTaskSlice(
                    connection=worker,
                    task_id=request_id,
                    artifact_id=f"{request_id}:{worker.runtime_id}",
                    method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
                    row_start=partition.start,
                    row_end=partition.end,
                    effective_gflops=worker_gflops[worker.peer_id],
                )
            )
        return assignments

    def dispatch_spatial_convolution(
        self,
        *,
        request_id: str,
        output_channels: int,
        workers: list[RuntimePeerConnection],
        worker_hardware: list[WorkerHardwareCapability],
        max_channels_per_task: int | None = None,
    ) -> list[WorkerTaskSlice]:
        """Use this when one convolution request needs output-channel slices across workers.

        Args: request_id logical task id, output_channels total channels, workers live workers, worker_hardware spatial performance entries, max_channels_per_task optional chunk cap.
        Returns: Weighted channel slices, chunked further when one worker slice exceeds the per-task channel cap.
        """
        worker_gflops: dict[str, float] = {worker.peer_id: 0.0 for worker in workers}
        for hardware in worker_hardware:
            worker_gflops[hardware.worker_peer_id] = worker_gflops.get(hardware.worker_peer_id, 0.0) + hardware.effective_gflops

        schedulable_workers = [worker for worker in workers if worker_gflops.get(worker.peer_id, 0.0) > 0.0]
        if not schedulable_workers:
            return []

        partitions = partition_contiguous_range(
            0,
            output_channels,
            [worker_gflops[worker.peer_id] for worker in schedulable_workers],
        )

        assignments: list[WorkerTaskSlice] = []
        for partition, worker in zip(partitions, schedulable_workers):
            if partition.end <= partition.start:
                continue
            chunk_size = max_channels_per_task or (partition.end - partition.start)
            chunk_index = 0
            for start_oc in range(partition.start, partition.end, chunk_size):
                end_oc = min(start_oc + chunk_size, partition.end)
                assignments.append(
                    WorkerTaskSlice(
                        connection=worker,
                        task_id=request_id,
                        artifact_id=f"{request_id}:{worker.runtime_id}:{chunk_index}",
                        method=METHOD_SPATIAL_CONVOLUTION,
                        row_start=0,
                        row_end=0,
                        start_oc=start_oc,
                        end_oc=end_oc,
                        effective_gflops=worker_gflops[worker.peer_id],
                    )
                )
                chunk_index += 1
        return assignments
