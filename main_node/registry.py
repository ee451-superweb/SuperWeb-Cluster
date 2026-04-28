"""Track connected runtime peers and their abstract scheduling capabilities.

Use this module when the main node needs a thread-safe view of registered
workers, registered clients, active tasks, active client requests, and the
effective GFLOPS values used for dispatch decisions.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from core.types import ComputePerformanceSummary, HardwareProfile, MethodPerformanceSummary
from core.constants import (
    METHOD_GEMV,
    METHOD_CONV2D,
    RUNTIME_ROLE_CLIENT,
    RUNTIME_ROLE_WORKER,
)
from core.tracing import trace_function
from main_node.mailbox import RuntimeConnectionMailbox


@dataclass(slots=True)
class RuntimePeerConnection:
    """Store one live worker or client connection plus its runtime state."""

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
    heartbeat_failure_count: int = 0
    last_request_at: float = 0.0
    active_request_id: str = ""
    active_task_id: str = ""
    active_method: str = ""
    io_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    task_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    mailbox: RuntimeConnectionMailbox = field(default_factory=RuntimeConnectionMailbox, repr=False, compare=False)


@dataclass(slots=True)
class WorkerHardwareCapability:
    """Represent one per-method performance entry attached to a worker."""

    hardware_id: str
    worker_peer_id: str
    worker_runtime_id: str
    worker_node_name: str
    method: str
    hardware_type: str
    effective_gflops: float
    rank: int


class ClusterRegistry:
    """Own the thread-safe worker/client registry used by the main node."""

    @trace_function
    def __init__(self) -> None:
        """Create empty worker, client, and hardware-capability registries.

        Args: self registry instance being initialized.
        Returns: None after counters, locks, and storage dictionaries are ready.
        """
        self._workers: dict[str, RuntimePeerConnection] = {}
        self._clients: dict[str, RuntimePeerConnection] = {}
        self._worker_hardware: dict[str, WorkerHardwareCapability] = {}
        self._next_hardware_id = 1
        self._next_worker_id = 1
        self._next_client_id = 1
        self._next_task_id = 1
        self._total_effective_gflops = 0.0
        self._lock = threading.Lock()

    def _normalize_task_prefix(self, method: str) -> str:
        """Return a compact task-id prefix derived from the requested method."""
        prefix = (method or "task").strip().lower()
        return "".join(char for char in prefix if char.isalnum() or char in {"-", "_"}) or "task"

    @trace_function
    def allocate_task_id(self, method: str) -> str:
        """Use this when the main node accepts a client request and needs a cluster task id.

        Args: method requested client method used to build a readable task-id prefix.
        Returns: A main-node-assigned task id unique within the current process lifetime.
        """
        with self._lock:
            task_id = f"{self._normalize_task_prefix(method)}-{self._next_task_id}"
            self._next_task_id += 1
            return task_id

    def _remove_worker_hardware_locked(self, connection: RuntimePeerConnection) -> None:
        """Use this inside the registry lock before removing or replacing worker performance.

        Args: connection registered worker whose hardware capability rows should be removed.
        Returns: None after the worker's hardware ids and aggregated GFLOPS are cleared.
        """
        for hardware_id in connection.hardware_ids:
            worker_hardware = self._worker_hardware.pop(hardware_id, None)
            if worker_hardware is not None:
                self._total_effective_gflops -= worker_hardware.effective_gflops
        connection.hardware_ids.clear()

    def _apply_worker_performance_locked(
        self,
        connection: RuntimePeerConnection,
        *,
        peer_id: str,
        node_name: str,
        runtime_id: str,
        performance: ComputePerformanceSummary,
    ) -> None:
        """Use this inside the registry lock to rewrite one worker's performance entries.

        Args: connection worker record to update plus peer metadata and the new abstract performance summary.
        Returns: None after per-method hardware rows and total GFLOPS are rebuilt.
        """
        self._remove_worker_hardware_locked(connection)
        connection.performance = performance
        method_summaries = list(performance.method_summaries)
        if not method_summaries and performance.ranked_hardware:
            method_summaries = [
                MethodPerformanceSummary(
                    method=METHOD_GEMV,
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
        """Use this when a compute node finishes REGISTER_WORKER successfully.

        Args: node_name/address/port identify the peer, hardware/performance describe it, and sock is the live TCP socket.
        Returns: The stored RuntimePeerConnection with its assigned worker runtime id.
        """
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
            self._apply_worker_performance_locked(
                connection,
                peer_id=peer_id,
                node_name=node_name,
                runtime_id=runtime_id,
                performance=performance,
            )
        return connection

    @trace_function
    def register_client(
        self,
        node_name: str,
        peer_address: str,
        peer_port: int,
        sock: socket.socket,
    ) -> RuntimePeerConnection:
        """Use this when a client completes CLIENT_JOIN and needs a runtime id.

        Args: node_name/address/port identify the peer and sock is the live client TCP socket.
        Returns: The stored RuntimePeerConnection with its assigned client runtime id.
        """
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
        """Use this when one worker disconnects or must be evicted from the registry.

        Args: peer_id internal registry key for the worker connection.
        Returns: The removed worker connection, or None if it was already absent.
        """
        with self._lock:
            connection = self._workers.pop(peer_id, None)
            if connection is None:
                return None

            self._remove_worker_hardware_locked(connection)
            return connection

    @trace_function
    def remove_client(self, peer_id: str) -> RuntimePeerConnection | None:
        """Use this when one client disconnects or its session must be dropped.

        Args: peer_id internal registry key for the client connection.
        Returns: The removed client connection, or None if it was not registered.
        """
        with self._lock:
            return self._clients.pop(peer_id, None)

    @trace_function
    def remove(self, peer_id: str) -> RuntimePeerConnection | None:
        """Use this generic remover when the caller does not know the peer role up front.

        Args: peer_id internal registry key for either a worker or a client.
        Returns: The removed connection, or None if no registered peer matched.
        """
        with self._lock:
            connection = self._workers.pop(peer_id, None)
            if connection is not None:
                self._remove_worker_hardware_locked(connection)
                return connection
            return self._clients.pop(peer_id, None)

    @trace_function
    def list_workers(self) -> list[RuntimePeerConnection]:
        """Use this when dispatch or heartbeat code needs a worker snapshot.

        Args: self registry queried for worker connections.
        Returns: A new list containing the currently registered workers.
        """
        with self._lock:
            return list(self._workers.values())

    @trace_function
    def list_clients(self) -> list[RuntimePeerConnection]:
        """Use this when client-management code needs a client snapshot.

        Args: self registry queried for client connections.
        Returns: A new list containing the currently registered clients.
        """
        with self._lock:
            return list(self._clients.values())

    @trace_function
    def list_connections(self) -> list[RuntimePeerConnection]:
        """Use this when shutdown or diagnostics need all live runtime peers together.

        Args: self registry queried for every registered connection.
        Returns: A new list containing workers followed by clients.
        """
        with self._lock:
            return list(self._workers.values()) + list(self._clients.values())

    @trace_function
    def mark_heartbeat(self, peer_id: str, sent_at: float | None = None) -> None:
        """Use this when the main node sends a heartbeat to a worker or client.

        Args: peer_id registry key for the peer and sent_at optional timestamp override.
        Returns: None after the peer's last-heartbeat timestamp is updated when present.
        """
        if sent_at is None:
            sent_at = time.time()
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is None:
                connection = self._clients.get(peer_id)
            if connection is not None:
                connection.last_heartbeat_at = sent_at
                connection.heartbeat_failure_count = 0

    @trace_function
    def record_heartbeat_failure(self, peer_id: str) -> int | None:
        """Use this when one heartbeat round fails for a worker.

        Args: peer_id registry key for the worker whose liveness probe failed.
        Returns: The worker's updated consecutive heartbeat-failure count, or None when absent.
        """
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is None:
                return None
            connection.heartbeat_failure_count += 1
            return connection.heartbeat_failure_count

    @trace_function
    def get_heartbeat_failure_count(self, peer_id: str) -> int | None:
        """Use this when coordination code needs the worker's current heartbeat failure count.

        Args: peer_id registry key for the worker being queried.
        Returns: The current consecutive heartbeat-failure count, or None when absent.
        """
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is None:
                return None
            return connection.heartbeat_failure_count

    @trace_function
    def mark_client_request(self, peer_id: str, sent_at: float | None = None) -> None:
        """Use this when the main node observes traffic from a client connection.

        Args: peer_id registry key for the client and sent_at optional timestamp override.
        Returns: None after the client's last-request timestamp is updated when present.
        """
        if sent_at is None:
            sent_at = time.time()
        with self._lock:
            connection = self._clients.get(peer_id)
            if connection is not None:
                connection.last_request_at = sent_at

    @trace_function
    def mark_worker_task(self, peer_id: str, *, request_id: str, task_id: str, method: str) -> None:
        """Use this before dispatching a task to record the worker's active assignment.

        Args: peer_id worker registry key plus request_id, task_id, and method for the in-flight task.
        Returns: None after the worker's active request/task fields are updated.
        """
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is not None:
                connection.active_request_id = request_id
                connection.active_task_id = task_id
                connection.active_method = method

    @trace_function
    def clear_worker_task(self, peer_id: str, *, task_id: str | None = None) -> None:
        """Use this after a worker task completes or is abandoned.

        Args: peer_id worker registry key and task_id optional guard to avoid clearing the wrong task.
        Returns: None after matching active-task state is cleared.
        """
        with self._lock:
            connection = self._workers.get(peer_id)
            if connection is not None and (task_id is None or connection.active_task_id == task_id):
                connection.active_request_id = ""
                connection.active_task_id = ""
                connection.active_method = ""

    @trace_function
    def mark_client_request_state(self, peer_id: str, *, task_id: str, method: str) -> None:
        """Use this when one client begins an application-level request.

        Args: peer_id client registry key plus task_id and method now active for that client.
        Returns: None after the client's active-task fields are updated.
        """
        with self._lock:
            connection = self._clients.get(peer_id)
            if connection is not None:
                connection.active_request_id = task_id
                connection.active_task_id = task_id
                connection.active_method = method

    @trace_function
    def clear_client_request_state(self, peer_id: str, *, task_id: str | None = None) -> None:
        """Use this after a client request finishes or is cancelled.

        Args: peer_id client registry key and task_id optional guard for the active task.
        Returns: None after matching client-request state is cleared.
        """
        with self._lock:
            connection = self._clients.get(peer_id)
            if connection is not None and (task_id is None or connection.active_task_id == task_id):
                connection.active_request_id = ""
                connection.active_task_id = ""
                connection.active_method = ""

    @trace_function
    def clear(self) -> list[RuntimePeerConnection]:
        """Use this during shutdown to empty the entire runtime registry at once.

        Args: self registry being fully cleared.
        Returns: A list of all connections that were registered before the clear.
        """
        with self._lock:
            items = list(self._workers.values()) + list(self._clients.values())
            self._workers.clear()
            self._clients.clear()
            self._worker_hardware.clear()
            self._total_effective_gflops = 0.0
            return items

    @trace_function
    def count_workers(self) -> int:
        """Use this when replies or diagnostics need the current worker count.

        Args: self registry queried for registered worker count.
        Returns: The number of workers currently registered.
        """
        with self._lock:
            return len(self._workers)

    @trace_function
    def count_clients(self) -> int:
        """Use this when replies or diagnostics need the current client count.

        Args: self registry queried for registered client count.
        Returns: The number of clients currently registered.
        """
        with self._lock:
            return len(self._clients)

    @trace_function
    def count(self) -> int:
        """Use this when diagnostics need the total number of live runtime peers.

        Args: self registry queried for total connection count.
        Returns: The combined number of registered workers and clients.
        """
        with self._lock:
            return len(self._workers) + len(self._clients)

    @trace_function
    def list_worker_hardware(self, method: str | None = None) -> list[WorkerHardwareCapability]:
        """Use this when dispatch needs performance rows for all workers or one method.

        Args: method optional method filter; None keeps every registered performance row.
        Returns: A new list of worker capability rows, filtered when requested.
        """
        with self._lock:
            items = list(self._worker_hardware.values())
        if method is None:
            return items
        return [item for item in items if item.method == method]

    @trace_function
    def count_registered_hardware(self) -> int:
        """Use this for diagnostics that need the number of stored performance rows.

        Args: self registry queried for registered capability count.
        Returns: The number of stored WorkerHardwareCapability entries.
        """
        with self._lock:
            return len(self._worker_hardware)

    @trace_function
    def total_registered_gflops(self) -> float:
        """Use this when the main node wants a coarse cluster-capacity summary.

        Args: self registry queried for aggregated effective GFLOPS.
        Returns: The sum of registered effective GFLOPS across all capability rows.
        """
        with self._lock:
            return self._total_effective_gflops

    @trace_function
    def total_registered_gflops_by_method(self) -> dict[str, float]:
        """Use this when diagnostics need per-method effective GFLOPS totals.

        Args: self registry queried for per-method aggregate effective GFLOPS.
        Returns: A dictionary keyed by method name with summed effective GFLOPS.
        """
        totals = {
            METHOD_GEMV: 0.0,
            METHOD_CONV2D: 0.0,
        }
        with self._lock:
            for worker_hardware in self._worker_hardware.values():
                totals[worker_hardware.method] = (
                    totals.get(worker_hardware.method, 0.0) + worker_hardware.effective_gflops
                )
        return totals

    @trace_function
    def get_worker(self, peer_id: str) -> RuntimePeerConnection | None:
        """Use this when code already has a worker peer id and needs its connection record.

        Args: peer_id registry key for the worker connection.
        Returns: The matching worker connection, or None if absent.
        """
        with self._lock:
            return self._workers.get(peer_id)

    @trace_function
    def get_worker_by_runtime_id(self, runtime_id: str) -> RuntimePeerConnection | None:
        """Use this when protocol messages identify a worker by runtime id instead of peer id.

        Args: runtime_id main-node-assigned worker id such as ``worker-3``.
        Returns: The matching worker connection, or None if no worker has that id.
        """
        with self._lock:
            for connection in self._workers.values():
                if connection.runtime_id == runtime_id:
                    return connection
        return None

    @trace_function
    def update_worker_performance_by_runtime_id(
        self,
        runtime_id: str,
        performance: ComputePerformanceSummary,
    ) -> RuntimePeerConnection | None:
        """Use this after a worker sends WORKER_UPDATE with fresh performance data.

        Args: runtime_id target worker runtime id and performance replacement abstract summary.
        Returns: The updated worker connection, or None when the worker is unknown.
        """
        with self._lock:
            for peer_id, connection in self._workers.items():
                if connection.runtime_id != runtime_id:
                    continue
                self._apply_worker_performance_locked(
                    connection,
                    peer_id=peer_id,
                    node_name=connection.node_name,
                    runtime_id=runtime_id,
                    performance=performance,
                )
                return connection
        return None

    @trace_function
    def get_client_active_task_ids(self, peer_id: str) -> tuple[str, ...]:
        """Use this when answering CLIENT_INFO_REQUEST for one client session.

        Args: peer_id registry key for the client connection being queried.
        Returns: A tuple containing the current active task id, or an empty tuple when idle.
        """
        with self._lock:
            connection = self._clients.get(peer_id)
            if connection is None or not connection.active_task_id:
                return ()
            return (connection.active_task_id,)

ComputeNodeRegistry = ClusterRegistry
WorkerRegistry = ClusterRegistry

