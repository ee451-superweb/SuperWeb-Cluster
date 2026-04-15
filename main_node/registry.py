"""Registry of connected workers and clients."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from common.types import ComputePerformanceSummary, HardwareProfile, MethodPerformanceSummary
from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, RUNTIME_ROLE_CLIENT, RUNTIME_ROLE_WORKER
from app.trace_utils import trace_function


@dataclass(slots=True)
class RuntimePeerConnection:
    """Connected runtime peer metadata stored by the scheduler."""

    peer_id: str
    runtime_id: str
    node_name: str
    role: str
    peer_address: str
    peer_port: int
    sock: socket.socket = field(repr=False, compare=False)
    hardware: HardwareProfile | None = None
    performance: ComputePerformanceSummary | None = None
    hardware_ids: list[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = 0.0
    last_request_at: float = 0.0
    io_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


@dataclass(slots=True)
class WorkerHardwareCapability:
    """One benchmarked hardware backend attached to a registered worker."""

    hardware_id: str
    worker_peer_id: str
    worker_runtime_id: str
    worker_node_name: str
    method: str
    hardware_type: str
    effective_gflops: float
    rank: int


class ClusterRegistry:
    """Thread-safe pools of connected workers and clients."""

    @trace_function
    def __init__(self) -> None:
        self._workers: dict[str, RuntimePeerConnection] = {}
        self._clients: dict[str, RuntimePeerConnection] = {}
        self._worker_hardware: dict[str, WorkerHardwareCapability] = {}
        self._next_hardware_id = 1
        self._next_worker_id = 1
        self._next_client_id = 1
        self._total_effective_gflops = 0.0
        self._lock = threading.Lock()

    @trace_function
    def register_worker(
        self,
        node_name: str,
        peer_address: str,
        peer_port: int,
        hardware: HardwareProfile,
        performance: ComputePerformanceSummary,
        sock: socket.socket,
    ) -> RuntimePeerConnection:
        peer_id = f"worker:{node_name}@{peer_address}:{peer_port}"
        with self._lock:
            runtime_id = f"worker-{self._next_worker_id}"
            self._next_worker_id += 1
            connection = RuntimePeerConnection(
                peer_id=peer_id,
                runtime_id=runtime_id,
                node_name=node_name,
                role=RUNTIME_ROLE_WORKER,
                peer_address=peer_address,
                peer_port=peer_port,
                hardware=hardware,
                performance=performance,
                sock=sock,
            )
            self._workers[peer_id] = connection
            method_summaries = list(performance.method_summaries)
            if not method_summaries and performance.ranked_hardware:
                method_summaries = [
                    MethodPerformanceSummary(
                        method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
                        hardware_count=performance.hardware_count,
                        ranked_hardware=list(performance.ranked_hardware),
                    )
                ]

            for method_summary in method_summaries:
                for reported_hardware in method_summary.ranked_hardware:
                    hardware_id = f"hardware:{self._next_hardware_id}"
                    self._next_hardware_id += 1
                    worker_hardware = WorkerHardwareCapability(
                        hardware_id=hardware_id,
                        worker_peer_id=peer_id,
                        worker_runtime_id=runtime_id,
                        worker_node_name=node_name,
                        method=method_summary.method,
                        hardware_type=reported_hardware.hardware_type,
                        effective_gflops=reported_hardware.effective_gflops,
                        rank=reported_hardware.rank,
                    )
                    self._worker_hardware[hardware_id] = worker_hardware
                    connection.hardware_ids.append(hardware_id)
                    self._total_effective_gflops += reported_hardware.effective_gflops
        return connection

    @trace_function
    def register_client(
        self,
        node_name: str,
        peer_address: str,
        peer_port: int,
        sock: socket.socket,
    ) -> RuntimePeerConnection:
        peer_id = f"client:{node_name}@{peer_address}:{peer_port}"
        with self._lock:
            runtime_id = f"client-{self._next_client_id}"
            self._next_client_id += 1
            connection = RuntimePeerConnection(
                peer_id=peer_id,
                runtime_id=runtime_id,
                node_name=node_name,
                role=RUNTIME_ROLE_CLIENT,
                peer_address=peer_address,
                peer_port=peer_port,
                sock=sock,
            )
            self._clients[peer_id] = connection
        return connection

    @trace_function
    def remove_worker(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            connection = self._workers.pop(peer_id, None)
            if connection is None:
                return None

            for hardware_id in connection.hardware_ids:
                worker_hardware = self._worker_hardware.pop(hardware_id, None)
                if worker_hardware is not None:
                    self._total_effective_gflops -= worker_hardware.effective_gflops
            return connection

    @trace_function
    def remove_client(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            return self._clients.pop(peer_id, None)

    @trace_function
    def remove(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            connection = self._workers.pop(peer_id, None)
            if connection is not None:
                for hardware_id in connection.hardware_ids:
                    worker_hardware = self._worker_hardware.pop(hardware_id, None)
                    if worker_hardware is not None:
                        self._total_effective_gflops -= worker_hardware.effective_gflops
                return connection
            return self._clients.pop(peer_id, None)

    @trace_function
    def list_workers(self) -> list[RuntimePeerConnection]:
        with self._lock:
            return list(self._workers.values())

    @trace_function
    def list_clients(self) -> list[RuntimePeerConnection]:
        with self._lock:
            return list(self._clients.values())

    @trace_function
    def list_connections(self) -> list[RuntimePeerConnection]:
        with self._lock:
            return list(self._workers.values()) + list(self._clients.values())

    @trace_function
    def mark_heartbeat(self, peer_id: str, sent_at: float | None = None) -> None:
        if sent_at is None:
            sent_at = time.time()
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is not None:
                connection.last_heartbeat_at = sent_at

    @trace_function
    def mark_client_request(self, peer_id: str, sent_at: float | None = None) -> None:
        if sent_at is None:
            sent_at = time.time()
        with self._lock:
            connection = self._clients.get(peer_id)
            if connection is not None:
                connection.last_request_at = sent_at

    @trace_function
    def clear(self) -> list[RuntimePeerConnection]:
        with self._lock:
            items = list(self._workers.values()) + list(self._clients.values())
            self._workers.clear()
            self._clients.clear()
            self._worker_hardware.clear()
            self._total_effective_gflops = 0.0
            return items

    @trace_function
    def count_workers(self) -> int:
        with self._lock:
            return len(self._workers)

    @trace_function
    def count_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    @trace_function
    def count(self) -> int:
        with self._lock:
            return len(self._workers) + len(self._clients)

    @trace_function
    def list_worker_hardware(self, method: str | None = None) -> list[WorkerHardwareCapability]:
        with self._lock:
            items = list(self._worker_hardware.values())
        if method is None:
            return items
        return [item for item in items if item.method == method]

    @trace_function
    def count_registered_hardware(self) -> int:
        with self._lock:
            return len(self._worker_hardware)

    @trace_function
    def total_registered_gflops(self) -> float:
        with self._lock:
            return self._total_effective_gflops

    @trace_function
    def get_worker(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            return self._workers.get(peer_id)

ComputeNodeRegistry = ClusterRegistry
WorkerRegistry = ClusterRegistry

