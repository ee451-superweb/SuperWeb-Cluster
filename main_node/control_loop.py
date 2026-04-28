"""Run the main-node control plane for discovery, client sessions, and workers.

Use this module when one process is acting as the cluster's main node. It
accepts worker and client TCP sessions, answers discovery packets, dispatches
requests, emits heartbeats, and updates scheduling state from worker reports.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable

from adapters import network
from adapters.audit_log import write_audit_event
from core.types import DiscoveryResult
from compute_node.compute_methods.conv2d import (
    DEFAULT_DATASET_DIR as CONV2D_DATASET_DIR,
)
from compute_node.input_matrix import build_input_matrix_spec
from core.config import AppConfig
from core.constants import (
    MAIN_NODE_NAME,
    METHOD_GEMV,
    METHOD_CONV2D,
)
from discovery import multicast
from main_node import client_session_service as client_session_service_module
from main_node import connection_service as connection_service_module
from main_node import main_handlers as runtime_handlers_module
from main_node.aggregator import ResultAggregator
from main_node.client_session_service import ClientSessionService
from main_node.connection_service import RuntimeConnectionService
from main_node.dispatcher import TaskDispatcher, WorkerTaskSlice
from main_node.main_handlers import RuntimeConnectionHandler
from main_node.heartbeat import HeartbeatCoordinator
from main_node.request_handler import ClientRequestHandler
from main_node.registry import ClusterRegistry, RuntimePeerConnection
from main_node.task_exchange import WorkerTaskExchange
from transport.artifact_manager import ArtifactManager
from wire.internal_protocol.transport import (
    MessageKind,
    build_client_response,
    build_register_ok,
    describe_message_kind,
    recv_message,
    send_message,
)
from core.tracing import trace_function


class MainNodeRuntime:
    """Own the main-node runtime from startup through shutdown."""

    @trace_function
    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Create the services that the main node needs for runtime orchestration.

        Args: config runtime settings, logger diagnostics sink, should_stop optional
            shutdown predicate.
        Returns: None after the runtime wires registry, dispatcher, transfer, and session services together.
        """
        self.config = config
        self.logger = logger
        self.should_stop = should_stop or (lambda: False)
        self.registry = ClusterRegistry()
        self.dispatcher = TaskDispatcher()
        self.aggregator = ResultAggregator()
        self.gemv_spec = build_input_matrix_spec()
        self.conv2d_dataset_dir = CONV2D_DATASET_DIR
        self.artifact_manager = ArtifactManager(
            root_dir=self.conv2d_dataset_dir / "artifacts",
            public_host="127.0.0.1",
            port=self.config.data_plane_port,
            chunk_size=self.config.artifact_chunk_size,
        )
        self._stop_event = threading.Event()
        self.task_exchange = WorkerTaskExchange(
            config=self.config,
            conv2d_dataset_dir=self.conv2d_dataset_dir,
            remove_worker_connection=self._remove_worker_connection,
            artifact_manager=self.artifact_manager,
            logger=self.logger,
        )
        self.client_request_handler = ClientRequestHandler(
            config=self.config,
            registry=self.registry,
            dispatcher=self.dispatcher,
            aggregator=self.aggregator,
            gemv_spec=self.gemv_spec,
            conv2d_dataset_dir=self.conv2d_dataset_dir,
            task_exchange=self.task_exchange,
            artifact_manager=self.artifact_manager,
            cluster_counts=self._cluster_counts,
            logger=self.logger,
        )
        self.heartbeat_coordinator = HeartbeatCoordinator(
            config=self.config,
            registry=self.registry,
            remove_connection=self._remove_runtime_connection,
            logger=self.logger,
        )
        self.client_session_service = ClientSessionService(
            config=self.config,
            registry=self.registry,
            build_client_response_for_request=self._build_client_response_for_request,
            allocate_data_plane_endpoints=self._allocate_data_plane_endpoints,
            remove_client_connection=self._remove_client_connection,
            logger=self.logger,
        )
        self.connection_service = RuntimeConnectionService(
            config=self.config,
            registry=self.registry,
            format_reported_hardware=self._format_reported_hardware,
            print_cluster_compute_capacity=self._print_cluster_compute_capacity,
            serve_client_connection=self._serve_client_connection,
            cluster_counts=self._cluster_counts,
            on_runtime_connection_registered=self._start_runtime_connection_reader,
            logger=self.logger,
        )
        self.runtime_connection_handler = RuntimeConnectionHandler(
            config=self.config,
            registry=self.registry,
            format_reported_hardware=self._format_reported_hardware,
            print_cluster_compute_capacity=self._print_cluster_compute_capacity,
            runtime_should_stop=self._runtime_should_stop,
            logger=self.logger,
        )

    @trace_function
    def _print_startup_banner(self, local_ip: str, local_mac: str) -> None:
        """Print the information that identifies this process as the main node."""

        startup_mode = "announce" if self.config.role == "announce" else "promoted"
        write_audit_event(
            f"started as main node mode={startup_mode} endpoint={local_ip}:{self.config.tcp_port}",
            stdout=True,
            logger=self.logger,
        )
        if self.config.role == "announce":
            self.logger.info("Starting main-node runtime.")
        else:
            self.logger.info("No main node discovered after retry limit. Promoting self to main node.")
        self.logger.info("main_node_ip=%s", local_ip)
        self.logger.info("main_node_mac=%s", local_mac)
        self.logger.info("main_node_tcp_port=%s", self.config.tcp_port)
        self.logger.info("main_node_data_plane_port=%s", self.artifact_manager.port)
        self.logger.info(
            "Listening for superweb-cluster mDNS discovery on %s:%s",
            self.config.multicast_group,
            self.config.udp_port,
        )
        self.logger.info(
            "Listening for superweb-cluster runtime connections on 0.0.0.0:%s",
            self.config.tcp_port,
        )
        self.logger.info(
            "Listening for superweb-cluster artifact data plane on 0.0.0.0:%s",
            self.artifact_manager.port,
        )

    def _runtime_should_stop(self) -> bool:
        """Use this helper wherever long-running loops need a shared stop condition.

        Args: self runtime instance being queried for shutdown state.
        Returns: True when either the internal stop event or external predicate says to stop.
        """
        return self._stop_event.is_set() or self.should_stop()

    def _refresh_runtime_service_bindings(self) -> None:
        """Refresh service references so tests and runtime code share the latest state.

        Use this before delegating to helper services because tests often swap
        ``runtime.registry`` or patched transport helpers after construction.

        Args:
            None.

        Returns:
            ``None`` after services and helper modules point at current bindings.
        """
        self.connection_service.registry = self.registry
        self.connection_service.logger = self.logger
        self.client_session_service.registry = self.registry
        self.client_session_service.logger = self.logger
        self.heartbeat_coordinator.registry = self.registry
        self.heartbeat_coordinator.logger = self.logger
        self.client_request_handler.registry = self.registry
        self.client_request_handler.logger = self.logger
        self.task_exchange.logger = self.logger
        self.runtime_connection_handler.registry = self.registry
        self.runtime_connection_handler.logger = self.logger
        self.connection_service.serve_client_connection = self._serve_client_connection
        self.runtime_connection_handler.runtime_should_stop = self._runtime_should_stop
        self.runtime_connection_handler.format_reported_hardware = self._format_reported_hardware
        self.runtime_connection_handler.print_cluster_compute_capacity = self._print_cluster_compute_capacity

        connection_service_module.recv_message = recv_message
        connection_service_module.send_message = send_message
        connection_service_module.build_register_ok = build_register_ok
        connection_service_module.build_client_response = build_client_response
        connection_service_module.MessageKind = MessageKind
        connection_service_module.describe_message_kind = describe_message_kind

        client_session_service_module.recv_message = recv_message
        client_session_service_module.send_message = send_message
        client_session_service_module.MessageKind = MessageKind
        client_session_service_module.describe_message_kind = describe_message_kind

        runtime_handlers_module.send_message = send_message
        runtime_handlers_module.recv_message = recv_message

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
        """Use this when replies or logs need current worker/client totals.

        Args: self runtime instance providing access to the cluster registry.
        Returns: A ``(worker_count, client_count)`` tuple.
        """
        return self.registry.count_workers(), self.registry.count_clients()

    def _method_display_name(self, method: str) -> str:
        """Return a short operator-facing label for one method name."""

        if method == METHOD_GEMV:
            return "gemv"
        if method == METHOD_CONV2D:
            return "conv2d"
        return method

    def _format_reported_hardware(self, connection: RuntimePeerConnection) -> str:
        """Use this for logs that should summarize a worker's abstract performance rows.

        Args: connection registered worker/client connection whose performance should be formatted.
        Returns: A compact string listing per-method hardware names and effective GFLOPS.
        """
        if connection.performance is None:
            return "[]"
        if connection.performance.method_summaries:
            entries: list[str] = []
            for method_summary in connection.performance.method_summaries:
                method_name = self._method_display_name(method_summary.method)
                if not method_summary.ranked_hardware:
                    entries.append(f"{method_name}=<unavailable>")
                    continue
                entries.append(
                    f"{method_name}="
                    + ",".join(
                        f"{item.hardware_type}:{item.effective_gflops:.3f}GFLOPS"
                        for item in method_summary.ranked_hardware
                    )
                )
            return "[" + ", ".join(entries) + "]"
        if not connection.performance.ranked_hardware:
            return "[]"
        return "[" + ", ".join(
            f"{item.hardware_type}:{item.effective_gflops:.3f}GFLOPS"
            for item in connection.performance.ranked_hardware
        ) + "]"

    def _print_cluster_compute_capacity(self) -> None:
        """Use this after registry changes to print a coarse cluster-capacity summary.

        Args: self runtime instance whose registry is being summarized.
        Returns: None after the summary line is printed to stdout.
        """
        method_totals = self.registry.total_registered_gflops_by_method()
        if not isinstance(method_totals, dict):
            method_totals = {}

        def _safe_method_total(method: str) -> float:
            try:
                return float(method_totals.get(method, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        self.logger.info(
            "Current cluster compute capacity total_effective_gflops=%.3f worker_count=%s "
            "hardware_count=%s gemv_effective_gflops=%.3f conv2d_effective_gflops=%.3f",
            self.registry.total_registered_gflops(),
            self.registry.count_workers(),
            self.registry.count_registered_hardware(),
            _safe_method_total(METHOD_GEMV),
            _safe_method_total(METHOD_CONV2D),
        )

    @trace_function
    def _register_worker_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
        register_worker,
    ) -> RuntimePeerConnection:
        """Use this helper when a REGISTER_WORKER handshake needs connection-service wiring.

        Args: client_sock/addr accepted TCP socket tuple, main_node_ip listener address, register_worker decoded handshake payload.
        Returns: The registered worker connection object created by the connection service.
        """
        self._refresh_runtime_service_bindings()
        return self.connection_service.register_worker_connection(client_sock, addr, main_node_ip, register_worker)

    @trace_function
    def _register_client_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        client_join,
    ) -> RuntimePeerConnection:
        """Use this helper when a CLIENT_JOIN handshake needs connection-service wiring.

        Args: client_sock/addr accepted TCP socket tuple and client_join decoded join payload.
        Returns: The registered client connection object created by the connection service.
        """
        self._refresh_runtime_service_bindings()
        return self.connection_service.register_client_connection(client_sock, addr, client_join)

    def _start_runtime_connection_reader(self, connection: RuntimePeerConnection) -> None:
        """Use this after registration so one background thread can read that socket.

        Args: connection registered runtime peer whose mailbox reader should be started.
        Returns: None after the daemon reader thread has been launched.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.start_runtime_connection_reader(connection)

    def _runtime_connection_reader_loop(self, connection: RuntimePeerConnection) -> None:
        """Use this background loop to keep one runtime socket drained into its mailbox.

        Args: connection registered runtime peer whose socket should be continuously read.
        Returns: None after the connection closes or the runtime stops.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.runtime_connection_reader_loop(connection)

    @trace_function
    def _register_runtime_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
    ) -> RuntimePeerConnection:
        """Use this generic helper when one accepted socket must identify itself first.

        Args: client_sock/addr accepted TCP socket tuple and main_node_ip listener address to advertise back.
        Returns: The registered worker or client connection selected by the handshake kind.
        """
        self._refresh_runtime_service_bindings()
        return self.connection_service.register_runtime_connection(client_sock, addr, main_node_ip)

    @trace_function
    def _accept_runtime_connections(self, server_sock: socket.socket, main_node_ip: str) -> None:
        """Use this background loop to accept and register new runtime TCP sessions.

        Args: server_sock main listening TCP socket and main_node_ip advertised host address.
        Returns: None after the accept loop exits.
        """
        self._refresh_runtime_service_bindings()
        self.connection_service.accept_runtime_connections(
            server_sock,
            main_node_ip,
            runtime_should_stop=self._runtime_should_stop,
        )

    @trace_function
    def _remove_worker_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        """Use this when a worker disconnects, times out, or must be removed.

        Args: connection worker connection to remove and reason human-readable removal cause.
        Returns: None after the registry, mailbox, socket, and logs are updated.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.remove_worker_connection(connection, reason)

    def _handle_worker_update(self, connection: RuntimePeerConnection, worker_update) -> None:
        """Use this when a worker sends updated abstract performance data.

        Args: connection worker that sent the update and worker_update decoded WORKER_UPDATE payload.
        Returns: None after the registry is updated and cluster capacity is reprinted.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.handle_worker_update(connection, worker_update)

    def _handle_client_info_request(self, connection: RuntimePeerConnection, client_info_request) -> None:
        """Use this when a client asks whether it still has active work on the main node.

        Args: connection client session and client_info_request decoded polling payload.
        Returns: None after one CLIENT_INFO_REPLY has been sent on that connection.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.handle_client_info_request(connection, client_info_request)

    def _ensure_conv2d_dataset_ready(self) -> None:
        """Use this before convolution dispatch so the shared dataset files are ready.

        Args: self runtime instance delegating dataset preparation to task exchange.
        Returns: None after required conv2d dataset files are ensured on disk.
        """
        self.task_exchange.ensure_conv2d_dataset_ready()

    def _conv2d_weight_slice(self, *, variant: str, spec, start_oc: int, end_oc: int) -> bytes:
        """Use this when one conv2d task needs only a slice of the full weight tensor.

        Args: variant workload name, spec convolution spec, and start_oc/end_oc output-channel bounds.
        Returns: The serialized weight bytes for the requested output-channel slice.
        """
        return self.task_exchange.conv2d_weight_slice(
            variant=variant,
            spec=spec,
            start_oc=start_oc,
            end_oc=end_oc,
        )

    @trace_function
    def _run_worker_task_slice(self, request, assignment: WorkerTaskSlice):
        """Use this wrapper when one prepared assignment should run against its target worker.

        Args: request original client request and assignment one worker slice produced by the dispatcher.
        Returns: The completed worker task result returned by task exchange.
        """
        self.registry.mark_worker_task(
            assignment.connection.peer_id,
            request_id=request.request_id,
            task_id=assignment.task_id,
            method=request.method,
        )
        try:
            return self.task_exchange.run_worker_task_slice(request, assignment)
        finally:
            self.registry.clear_worker_task(
                assignment.connection.peer_id,
                task_id=assignment.task_id,
            )

    @trace_function
    def _build_client_response_for_request(self, request, *, allocation=None):
        """Use this to delegate one client request into the request-handler pipeline.

        Args: request decoded client request payload from the client session service,
            allocation optional data-plane allocation registered before REQUEST_OK.
        Returns: A ready-to-send client response envelope.
        """
        self._rebind_request_handler_services()
        return self.client_request_handler.build_client_response_for_request(
            request,
            allocation=allocation,
        )

    @trace_function
    def _allocate_data_plane_endpoints(self, request):
        """Pre-register upload/download IDs on the artifact manager for one request.

        Args: request decoded client request envelope.
        Returns: A DataPlaneAllocation from the client request handler.
        """
        self._rebind_request_handler_services()
        return self.client_request_handler.allocate_data_plane_endpoints(request)

    def _rebind_request_handler_services(self) -> None:
        """Refresh mutable service references on the request handler.

        Some tests swap out runtime collaborators after construction, so before
        dispatching a request we copy the current bindings onto the handler.
        """
        self.client_request_handler.registry = self.registry
        self.client_request_handler.dispatcher = self.dispatcher
        self.client_request_handler.aggregator = self.aggregator
        self.client_request_handler.gemv_spec = self.gemv_spec
        self.client_request_handler.conv2d_dataset_dir = self.conv2d_dataset_dir
        self.client_request_handler.task_exchange = self.task_exchange
        self.client_request_handler.artifact_manager = self.artifact_manager
        self.client_request_handler.run_worker_task_slice = self._run_worker_task_slice

    @trace_function
    def _remove_client_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        """Use this when one client disconnects or its session must be forcibly removed.

        Args: connection client connection to remove and reason human-readable removal cause.
        Returns: None after the registry, mailbox, socket, and logs are updated.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.remove_client_connection(connection, reason)

    def _remove_runtime_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        """Use this role-aware helper when a runtime connection must be removed.

        Args: connection runtime peer to remove and reason human-readable removal cause.
        Returns: None after the connection is delegated to worker or client cleanup.
        """
        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.remove_runtime_connection(connection, reason)

    @trace_function
    def _serve_client_connection(self, connection: RuntimePeerConnection) -> None:
        """Process client requests on a registered client session."""

        self._refresh_runtime_service_bindings()
        self.client_session_service.build_client_response_for_request = self._build_client_response_for_request
        self.client_session_service.allocate_data_plane_endpoints = self._allocate_data_plane_endpoints
        self.client_session_service.remove_client_connection = self._remove_client_connection
        self.client_session_service.serve_client_connection(connection, self._runtime_should_stop)

    @trace_function
    def _await_heartbeat_ok(self, connection: RuntimePeerConnection, heartbeat_unix_time_ms: int) -> None:
        """Use this to wait for the acknowledgement of one previously sent heartbeat.

        Args: connection target peer and heartbeat_unix_time_ms timestamp used as the ack key.
        Returns: None after the heartbeat coordinator confirms the acknowledgement.
        """
        self.heartbeat_coordinator.registry = self.registry
        self.heartbeat_coordinator.await_heartbeat_ok(connection, heartbeat_unix_time_ms)

    @trace_function
    def _send_heartbeat_with_retry(self, connection: RuntimePeerConnection) -> None:
        """Use this when one connection needs heartbeat retry logic applied.

        Args: connection worker/client peer whose heartbeat should be sent with retries.
        Returns: None after the heartbeat coordinator finishes the send flow.
        """
        self.heartbeat_coordinator.registry = self.registry
        self.heartbeat_coordinator.send_heartbeat_with_retry(connection)

    @trace_function
    def _send_heartbeat_once(self) -> None:
        """Use this when the periodic heartbeat loop should fan out one heartbeat round.

        Args: self runtime instance delegating the send round to the heartbeat coordinator.
        Returns: None after one heartbeat cycle has been attempted.
        """
        self.heartbeat_coordinator.registry = self.registry
        self.heartbeat_coordinator.send_heartbeat_once()

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

        self._refresh_runtime_service_bindings()
        self.runtime_connection_handler.close_runtime_connections()

    @trace_function
    def _handle_packet(self, endpoint: multicast.MulticastSocket, addr: tuple[str, int], message: bytes) -> None:
        """Use this when one multicast packet may be a main-node discovery query.

        Args: endpoint bound multicast socket, addr sender address tuple, and message raw packet bytes.
        Returns: None after optionally sending one announce reply.
        """

        if not multicast.parse_discover_message(message):
            return

        description = multicast.describe_packet(message)
        self.logger.info("mDNS packet from %s:%s -> %s", addr[0], addr[1], description)
        announce_host = multicast.send_announce(endpoint, addr, self.config, MAIN_NODE_NAME)
        self.logger.info(
            "Sent main-node mDNS response using %s:%s",
            announce_host,
            self.config.tcp_port,
        )

    @trace_function
    def run(self) -> DiscoveryResult:
        """Use this as the main-node process entrypoint after configuration is ready.

        Args: self runtime instance containing network settings and shared services.
        Returns: A DiscoveryResult describing successful stop or startup/runtime failure.
        """

        runtime_sock = None
        try:
            endpoint = multicast.create_receiver(self.config)
        except OSError as exc:
            message = f"Unable to start main-node listener socket on UDP port {self.config.udp_port}: {exc}."
            if getattr(exc, "winerror", None) in {10013, 10048}:
                message += " The UDP port may already be in use on this machine; try a different --udp-port value."
            return DiscoveryResult(success=False, message=message)

        try:
            runtime_sock = self._create_tcp_listener()
        except OSError as exc:
            multicast.close(endpoint)
            message = f"Unable to start main-node TCP listener on port {self.config.tcp_port}: {exc}."
            if getattr(exc, "winerror", None) in {10013, 10048}:
                message += " The TCP port may already be in use on this machine; try a different --tcp-port value."
            return DiscoveryResult(success=False, message=message)

        try:
            self.artifact_manager.start()
        except OSError as exc:
            multicast.close(endpoint)
            network.safe_close(runtime_sock)
            message = f"Unable to start main-node data-plane listener on port {self.config.data_plane_port}: {exc}."
            if getattr(exc, "winerror", None) in {10013, 10048}:
                message += " The TCP port may already be in use on this machine; try a different --data-plane-port value."
            return DiscoveryResult(success=False, message=message)

        try:
            network.set_socket_timeout(endpoint.sock, self.config.main_node_poll_timeout)

            local_ip = network.resolve_local_ip()
            local_mac = network.get_local_mac_address()
            self.artifact_manager.set_public_host(local_ip)
            self._print_startup_banner(local_ip, local_mac)

            accept_thread = threading.Thread(
                target=self._accept_runtime_connections,
                args=(runtime_sock, local_ip),
                name="main-node-accept",
                daemon=True,
            )
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="main-node-heartbeat",
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
                source="main_node",
                message="Main-node runtime stopped.",
            )
        finally:
            self._stop_event.set()
            if runtime_sock is not None:
                network.safe_close(runtime_sock)
            self.artifact_manager.close()
            self._close_runtime_connections()
            multicast.close(endpoint)


