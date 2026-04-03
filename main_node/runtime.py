"""Home scheduler runtime loop for the kickoff version."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable

from adapters import network
from common.types import DiscoveryResult
from config import AppConfig
from constants import (
    HOME_SCHEDULER_NAME,
    RUNTIME_MSG_CLIENT_JOIN,
    RUNTIME_MSG_CLIENT_REQUEST,
    RUNTIME_MSG_CLIENT_RESPONSE,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_REGISTER_WORKER,
)
from discovery import multicast
from main_node.registry import HomeClusterRegistry, RuntimePeerConnection
from runtime_protocol import (
    MessageKind,
    build_client_response,
    build_heartbeat,
    build_register_ok,
    describe_message_kind,
    recv_message,
    send_message,
)
from trace_utils import trace_function


class MainNodeRuntime:
    """Home scheduler loop that listens for multicast and responds to discovery."""

    @trace_function
    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.should_stop = should_stop or (lambda: False)
        self.registry = HomeClusterRegistry()
        self._stop_event = threading.Event()

    @trace_function
    def _print_startup_banner(self, local_ip: str, local_mac: str) -> None:
        """Print the information that identifies this process as the home scheduler."""

        if self.config.role == "announce":
            print("Starting home scheduler runtime.", flush=True)
        else:
            print("No home scheduler discovered after retry limit. Promoting self to home scheduler.", flush=True)
        print(f"home_scheduler_ip={local_ip}", flush=True)
        print(f"home_scheduler_mac={local_mac}", flush=True)
        print(f"home_scheduler_tcp_port={self.config.tcp_port}", flush=True)
        print(
            f"Listening for home cluster mDNS discovery on {self.config.multicast_group}:{self.config.udp_port}",
            flush=True,
        )
        print(
            f"Listening for home cluster worker/client TCP runtime connections on 0.0.0.0:{self.config.tcp_port}",
            flush=True,
        )

    def _runtime_should_stop(self) -> bool:
        return self._stop_event.is_set() or self.should_stop()

    @trace_function
    def _create_tcp_listener(self) -> socket.socket:
        """Create the TCP listener used by workers and clients."""

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.config.tcp_port))
        sock.listen()
        sock.settimeout(self.config.main_node_poll_timeout)
        return sock

    def _cluster_counts(self) -> tuple[int, int]:
        return self.registry.count_workers(), self.registry.count_clients()

    @trace_function
    def _register_worker_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        scheduler_ip: str,
        register_worker,
    ) -> RuntimePeerConnection:
        payload = register_worker
        connection = self.registry.register_worker(
            node_name=payload.node_name,
            peer_address=addr[0],
            peer_port=addr[1],
            hardware=payload.hardware,
            sock=client_sock,
        )
        send_message(
            client_sock,
            build_register_ok(
                scheduler_ip=scheduler_ip,
                scheduler_port=self.config.tcp_port,
                scheduler_name=HOME_SCHEDULER_NAME,
            ),
        )
        print(
            f"Registered home computer {connection.node_name} from {connection.peer_address}:{connection.peer_port} "
            f"cpu={connection.hardware.logical_cpu_count} memory_bytes={connection.hardware.memory_bytes}",
            flush=True,
        )
        return connection

    @trace_function
    def _register_client_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        client_join,
    ) -> RuntimePeerConnection:
        connection = self.registry.register_client(
            node_name=client_join.client_name,
            peer_address=addr[0],
            peer_port=addr[1],
            sock=client_sock,
        )
        worker_count, client_count = self._cluster_counts()
        send_message(
            client_sock,
            build_client_response(
                request_id="join",
                ok=True,
                message=f"Client {connection.node_name} joined home scheduler.",
                payload="joined",
                worker_count=worker_count,
                client_count=client_count,
            ),
        )
        print(
            f"Registered home client {connection.node_name} from {connection.peer_address}:{connection.peer_port}",
            flush=True,
        )
        client_thread = threading.Thread(
            target=self._serve_client_connection,
            args=(connection,),
            name=f"home-client-{connection.node_name}",
            daemon=True,
        )
        client_thread.start()
        return connection

    @trace_function
    def _register_runtime_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        scheduler_ip: str,
    ) -> RuntimePeerConnection:
        """Receive and accept one worker or client registration."""

        client_sock.settimeout(self.config.runtime_socket_timeout)
        message = recv_message(client_sock, max_size=self.config.max_message_size)
        if message is None:
            raise ConnectionError("peer closed the TCP session before registration")

        if message.kind == MessageKind.REGISTER_WORKER and message.register_worker is not None:
            return self._register_worker_connection(client_sock, addr, scheduler_ip, message.register_worker)
        if message.kind == MessageKind.CLIENT_JOIN and message.client_join is not None:
            return self._register_client_connection(client_sock, addr, message.client_join)

        raise ValueError(
            f"expected {RUNTIME_MSG_REGISTER_WORKER} or {RUNTIME_MSG_CLIENT_JOIN}, got {describe_message_kind(message.kind)}"
        )

    @trace_function
    def _accept_runtime_connections(self, server_sock: socket.socket, scheduler_ip: str) -> None:
        """Accept worker and client TCP sessions in the background."""

        while not self._runtime_should_stop():
            try:
                client_sock, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._runtime_should_stop():
                    return
                raise

            try:
                self._register_runtime_connection(client_sock, addr, scheduler_ip)
            except (OSError, ValueError, ConnectionError) as exc:
                print(f"Rejected home cluster runtime connection from {addr[0]}:{addr[1]}: {exc}", flush=True)
                network.safe_close(client_sock)

    @trace_function
    def _build_client_response_for_request(self, request) -> object:
        return build_client_response(
            request_id=request.request_id or request.command,
            ok=True,
            message="",
            payload="",
            worker_count=0,
            client_count=0,
        )

    @trace_function
    def _remove_client_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        removed = self.registry.remove_client(connection.peer_id)
        if removed is not None:
            network.safe_close(removed.sock)
        print(
            f"Removed home client {connection.node_name} at {connection.peer_address}:{connection.peer_port}: {reason}",
            flush=True,
        )

    @trace_function
    def _serve_client_connection(self, connection: RuntimePeerConnection) -> None:
        """Process client requests on a registered client session."""

        while not self._runtime_should_stop():
            try:
                message = recv_message(connection.sock, max_size=self.config.max_message_size)
            except socket.timeout:
                continue
            except (OSError, ValueError, ConnectionError) as exc:
                self._remove_client_connection(connection, str(exc))
                return

            if message is None:
                self._remove_client_connection(connection, "client closed the TCP session")
                return

            if message.kind != MessageKind.CLIENT_REQUEST or message.client_request is None:
                print(
                    f"Ignoring unexpected client runtime message from {connection.node_name}: "
                    f"{describe_message_kind(message.kind)}",
                    flush=True,
                )
                continue

            request = message.client_request
            self.registry.mark_client_request(connection.peer_id)
            print(
                f"{RUNTIME_MSG_CLIENT_REQUEST} from {request.client_name} "
                f"request_id={request.request_id} command={request.command} payload={request.payload!r}",
                flush=True,
            )
            response = self._build_client_response_for_request(request)
            send_message(connection.sock, response)
            print(
                f"{RUNTIME_MSG_CLIENT_RESPONSE} to {request.client_name} "
                f"request_id={request.request_id or request.command}",
                flush=True,
            )

    @trace_function
    def _await_heartbeat_ok(self, connection: RuntimePeerConnection, heartbeat_unix_time_ms: int) -> None:
        """Wait for a matching heartbeat acknowledgement from a worker."""

        message = recv_message(connection.sock, max_size=self.config.max_message_size)
        if message is None:
            raise ConnectionError("worker closed the TCP session during heartbeat")
        if message.kind != MessageKind.HEARTBEAT_OK or message.heartbeat_ok is None:
            raise ValueError(f"expected {RUNTIME_MSG_HEARTBEAT_OK}, got {describe_message_kind(message.kind)}")
        if message.heartbeat_ok.heartbeat_unix_time_ms != heartbeat_unix_time_ms:
            raise ValueError(
                "received HEARTBEAT_OK for unexpected heartbeat timestamp "
                f"{message.heartbeat_ok.heartbeat_unix_time_ms}"
            )

        ack_time = (
            message.heartbeat_ok.received_unix_time_ms / 1000 if message.heartbeat_ok.received_unix_time_ms else time.time()
        )
        self.registry.mark_heartbeat(connection.peer_id, sent_at=ack_time)
        print(
            f"{RUNTIME_MSG_HEARTBEAT_OK} from {message.heartbeat_ok.node_name} for {heartbeat_unix_time_ms}",
            flush=True,
        )

    @trace_function
    def _send_heartbeat_with_retry(self, connection: RuntimePeerConnection) -> None:
        """Send heartbeat retries and remove dead workers when they stop replying."""

        total_attempts = self.config.heartbeat_retry_count + 1
        last_error: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            heartbeat = build_heartbeat(HOME_SCHEDULER_NAME)
            assert heartbeat.heartbeat is not None
            heartbeat_unix_time_ms = heartbeat.heartbeat.unix_time_ms

            try:
                send_message(connection.sock, heartbeat)
                print(
                    f"{RUNTIME_MSG_HEARTBEAT} to {connection.node_name} "
                    f"at {connection.peer_address}:{connection.peer_port} attempt={attempt}/{total_attempts}",
                    flush=True,
                )
                self._await_heartbeat_ok(connection, heartbeat_unix_time_ms)
                return
            except (socket.timeout, OSError, ConnectionError, ValueError) as exc:
                last_error = exc
                if attempt < total_attempts:
                    print(
                        f"{RUNTIME_MSG_HEARTBEAT} retry for {connection.node_name} "
                        f"at {connection.peer_address}:{connection.peer_port} after {exc}",
                        flush=True,
                    )

        removed = self.registry.remove_worker(connection.peer_id)
        if removed is not None:
            network.safe_close(removed.sock)
        print(
            f"Removed home computer {connection.node_name} at {connection.peer_address}:{connection.peer_port} "
            f"after heartbeat timeout: {last_error}",
            flush=True,
        )

    @trace_function
    def _send_heartbeat_once(self) -> None:
        """Send one scheduler heartbeat cycle to all registered workers."""

        for connection in self.registry.list_workers():
            self._send_heartbeat_with_retry(connection)

    @trace_function
    def _heartbeat_loop(self) -> None:
        """Periodically emit heartbeat messages on active worker sessions."""

        while not self._runtime_should_stop():
            deadline = time.monotonic() + self.config.heartbeat_interval
            while not self._runtime_should_stop():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.25, remaining))

            if self._runtime_should_stop():
                return

            self._send_heartbeat_once()

    @trace_function
    def _close_runtime_connections(self) -> None:
        """Best-effort close of all worker and client sessions."""

        for connection in self.registry.clear():
            network.safe_close(connection.sock)

    @trace_function
    def _handle_packet(self, endpoint: multicast.MulticastSocket, addr: tuple[str, int], message: bytes) -> None:
        """Reply to home scheduler browse queries."""

        if not multicast.parse_discover_message(message):
            return

        description = multicast.describe_packet(message)
        print(f"mDNS packet from {addr[0]}:{addr[1]} -> {description}", flush=True)
        announce_host = multicast.send_announce(endpoint, addr, self.config, HOME_SCHEDULER_NAME)
        print(f"Sent home scheduler mDNS response using {announce_host}:{self.config.tcp_port}", flush=True)

    @trace_function
    def run(self) -> DiscoveryResult:
        """Run the home scheduler multicast loop until shutdown is requested."""

        runtime_sock = None
        try:
            endpoint = multicast.create_receiver(self.config)
        except OSError as exc:
            message = f"Unable to start home scheduler listener socket on UDP port {self.config.udp_port}: {exc}."
            if getattr(exc, "winerror", None) in {10013, 10048}:
                message += " The UDP port may already be in use on this machine; try a different --udp-port value."
            return DiscoveryResult(success=False, message=message)

        try:
            runtime_sock = self._create_tcp_listener()
        except OSError as exc:
            multicast.close(endpoint)
            message = f"Unable to start home scheduler TCP listener on port {self.config.tcp_port}: {exc}."
            if getattr(exc, "winerror", None) in {10013, 10048}:
                message += " The TCP port may already be in use on this machine; try a different --tcp-port value."
            return DiscoveryResult(success=False, message=message)

        try:
            network.set_socket_timeout(endpoint.sock, self.config.main_node_poll_timeout)

            local_ip = network.resolve_local_ip()
            local_mac = network.get_local_mac_address()
            self._print_startup_banner(local_ip, local_mac)

            accept_thread = threading.Thread(
                target=self._accept_runtime_connections,
                args=(runtime_sock, local_ip),
                name="home-scheduler-accept",
                daemon=True,
            )
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="home-scheduler-heartbeat",
                daemon=True,
            )
            accept_thread.start()
            heartbeat_thread.start()

            while not self._runtime_should_stop():
                packet = multicast.recv_packet(endpoint, self.config.buffer_size)
                if packet is None:
                    continue

                addr, message = packet
                self._handle_packet(endpoint, addr, message)

            return DiscoveryResult(
                success=True,
                peer_address=local_ip,
                peer_port=self.config.tcp_port,
                source="home_scheduler",
                message="Home scheduler runtime stopped.",
            )
        finally:
            self._stop_event.set()
            if runtime_sock is not None:
                network.safe_close(runtime_sock)
            self._close_runtime_connections()
            multicast.close(endpoint)