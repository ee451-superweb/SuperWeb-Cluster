"""Registry of connected workers and clients."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from common.types import HardwareProfile
from constants import RUNTIME_ROLE_CLIENT, RUNTIME_ROLE_WORKER
from trace_utils import trace_function


@dataclass(slots=True)
class RuntimePeerConnection:
    """Connected runtime peer metadata stored by the scheduler."""

    peer_id: str
    node_name: str
    role: str
    peer_address: str
    peer_port: int
    sock: socket.socket = field(repr=False, compare=False)
    hardware: HardwareProfile | None = None
    registered_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = 0.0
    last_request_at: float = 0.0


class HomeClusterRegistry:
    """Thread-safe pools of connected workers and clients."""

    @trace_function
    def __init__(self) -> None:
        self._workers: dict[str, RuntimePeerConnection] = {}
        self._clients: dict[str, RuntimePeerConnection] = {}
        self._lock = threading.Lock()

    @trace_function
    def register_worker(
        self,
        node_name: str,
        peer_address: str,
        peer_port: int,
        hardware: HardwareProfile,
        sock: socket.socket,
    ) -> RuntimePeerConnection:
        peer_id = f"worker:{node_name}@{peer_address}:{peer_port}"
        connection = RuntimePeerConnection(
            peer_id=peer_id,
            node_name=node_name,
            role=RUNTIME_ROLE_WORKER,
            peer_address=peer_address,
            peer_port=peer_port,
            hardware=hardware,
            sock=sock,
        )
        with self._lock:
            self._workers[peer_id] = connection
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
        connection = RuntimePeerConnection(
            peer_id=peer_id,
            node_name=node_name,
            role=RUNTIME_ROLE_CLIENT,
            peer_address=peer_address,
            peer_port=peer_port,
            sock=sock,
        )
        with self._lock:
            self._clients[peer_id] = connection
        return connection

    @trace_function
    def remove_worker(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            return self._workers.pop(peer_id, None)

    @trace_function
    def remove_client(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            return self._clients.pop(peer_id, None)

    @trace_function
    def remove(self, peer_id: str) -> RuntimePeerConnection | None:
        with self._lock:
            connection = self._workers.pop(peer_id, None)
            if connection is not None:
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


HomeComputerRegistry = HomeClusterRegistry
WorkerRegistry = HomeClusterRegistry