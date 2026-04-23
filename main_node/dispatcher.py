"""Plan worker slices for one request from abstract per-method performance data.

Use this module when the main node has already accepted a client request and
needs to turn registered worker GFLOPS rankings into contiguous GEMV row ranges
or conv2d output-channel ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.constants import METHOD_GEMM, METHOD_GEMV, METHOD_CONV2D
from core.work_partition import partition_contiguous_range
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
    method: str = METHOD_GEMV
    start_oc: int = 0
    end_oc: int = 0
    m_start: int = 0
    m_end: int = 0


class TaskDispatcher:
    """Convert registered worker performance into weighted request partitions."""

    def dispatch_gemv(
        self,
        *,
        request_id: str,
        rows: int,
        workers: list[RuntimePeerConnection],
        worker_hardware: list[WorkerHardwareCapability],
    ) -> list[WorkerTaskSlice]:
        """Use this when one GEMV request needs row partitions across active workers.

        Args: request_id logical task id, rows total matrix rows, workers live workers, worker_hardware GEMV performance entries.
        Returns: Weighted row slices, or an empty list when no worker has usable GEMV capacity.
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
                    method=METHOD_GEMV,
                    row_start=partition.start,
                    row_end=partition.end,
                    effective_gflops=worker_gflops[worker.peer_id],
                )
            )
        return assignments

    def dispatch_gemm(
        self,
        *,
        request_id: str,
        rows: int,
        workers: list[RuntimePeerConnection],
        worker_hardware: list[WorkerHardwareCapability],
    ) -> list[WorkerTaskSlice]:
        """Use this when one GEMM request needs M-axis partitions across workers.

        Args: request_id logical task id, rows total M dimension, workers live
        workers, worker_hardware GEMM performance entries.
        Returns: Weighted M-axis slices, or an empty list when no worker has
        usable GEMM capacity (i.e. no cuBLAS-capable host in the cluster).
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
                    method=METHOD_GEMM,
                    row_start=0,
                    row_end=0,
                    m_start=partition.start,
                    m_end=partition.end,
                    effective_gflops=worker_gflops[worker.peer_id],
                )
            )
        return assignments

    def dispatch_conv2d(
        self,
        *,
        request_id: str,
        output_channels: int,
        workers: list[RuntimePeerConnection],
        worker_hardware: list[WorkerHardwareCapability],
    ) -> list[WorkerTaskSlice]:
        """Use this when one convolution request needs output-channel slices across workers.

        Args: request_id logical task id, output_channels total channels, workers live workers, worker_hardware spatial performance entries.
        Returns: One weighted channel slice per schedulable worker.
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
            assignments.append(
                WorkerTaskSlice(
                    connection=worker,
                    task_id=request_id,
                    artifact_id=f"{request_id}:{worker.runtime_id}:0",
                    method=METHOD_CONV2D,
                    row_start=0,
                    row_end=0,
                    start_oc=partition.start,
                    end_oc=partition.end,
                    effective_gflops=worker_gflops[worker.peer_id],
                )
            )
        return assignments
