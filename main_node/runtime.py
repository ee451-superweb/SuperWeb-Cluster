"""Main-node runtime loop for superweb-cluster Sprint 1."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path

from adapters import network
from common.types import DiscoveryResult
from compute_node.compute_methods.spatial_convolution import (
    DATASET_GENERATE_SCRIPT_PATH as SPATIAL_DATASET_GENERATE_SCRIPT_PATH,
    DEFAULT_DATASET_DIR as SPATIAL_DATASET_DIR,
)
from compute_node.compute_methods.spatial_convolution.executor import load_named_workload_spec
from compute_node.input_matrix import build_input_matrix_spec
from app.config import AppConfig
from app.constants import (
    MAIN_NODE_NAME,
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
    METHOD_SPATIAL_CONVOLUTION,
    RUNTIME_MSG_CLIENT_JOIN,
    RUNTIME_MSG_CLIENT_REQUEST,
    RUNTIME_MSG_CLIENT_RESPONSE,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_REGISTER_WORKER,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
)
from discovery import multicast
from main_node.aggregator import ResultAggregator
from main_node.dispatcher import TaskDispatcher, WorkerTaskSlice
from main_node.registry import ClusterRegistry, RuntimePeerConnection
from wire.runtime import (
    MessageKind,
    build_client_response,
    build_heartbeat,
    build_register_ok,
    build_task_assign,
    describe_message_kind,
    recv_message,
    send_message,
)
from app.trace_utils import trace_function


class MainNodeRuntime:
    """Main-node loop that listens for multicast and responds to discovery."""

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
        self.registry = ClusterRegistry()
        self.dispatcher = TaskDispatcher()
        self.aggregator = ResultAggregator()
        self.fixed_matvec_spec = build_input_matrix_spec()
        self.spatial_dataset_dir = SPATIAL_DATASET_DIR
        self._stop_event = threading.Event()

    @trace_function
    def _print_startup_banner(self, local_ip: str, local_mac: str) -> None:
        """Print the information that identifies this process as the main node."""

        if self.config.role == "announce":
            print("Starting main-node runtime.", flush=True)
        else:
            print("No main node discovered after retry limit. Promoting self to main node.", flush=True)
        print(f"main_node_ip={local_ip}", flush=True)
        print(f"main_node_mac={local_mac}", flush=True)
        print(f"main_node_tcp_port={self.config.tcp_port}", flush=True)
        print(
            f"Listening for superweb-cluster mDNS discovery on {self.config.multicast_group}:{self.config.udp_port}",
            flush=True,
        )
        print(
            f"Listening for superweb-cluster runtime connections on 0.0.0.0:{self.config.tcp_port}",
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

    def _format_reported_hardware(self, connection: RuntimePeerConnection) -> str:
        if connection.performance is None:
            return "[]"
        if connection.performance.method_summaries:
            entries: list[str] = []
            for method_summary in connection.performance.method_summaries:
                for item in method_summary.ranked_hardware:
                    entries.append(f"{method_summary.method}:{item.hardware_type}:{item.effective_gflops:.3f}")
            return "[" + ", ".join(entries) + "]"
        if not connection.performance.ranked_hardware:
            return "[]"
        return "[" + ", ".join(
            f"{item.hardware_type}:{item.effective_gflops:.3f}"
            for item in connection.performance.ranked_hardware
        ) + "]"

    def _print_cluster_compute_capacity(self) -> None:
        print(
            "Current cluster compute capacity "
            f"total_effective_gflops={self.registry.total_registered_gflops():.3f} "
            f"worker_count={self.registry.count_workers()} "
            f"hardware_count={self.registry.count_registered_hardware()}",
            flush=True,
        )

    @trace_function
    def _register_worker_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
        register_worker,
    ) -> RuntimePeerConnection:
        payload = register_worker
        connection = self.registry.register_worker(
            node_name=payload.node_name,
            peer_address=addr[0],
            peer_port=addr[1],
            hardware=payload.hardware,
            performance=payload.performance,
            sock=client_sock,
        )
        send_message(
            client_sock,
            build_register_ok(
                main_node_ip=main_node_ip,
                main_node_port=self.config.tcp_port,
                main_node_name=MAIN_NODE_NAME,
                node_id=connection.runtime_id,
            ),
        )
        print(
            f"Registered compute node {connection.node_name} "
            f"id={connection.runtime_id} "
            f"from {connection.peer_address}:{connection.peer_port} "
            f"cpu={connection.hardware.logical_cpu_count} memory_bytes={connection.hardware.memory_bytes} "
            f"reported_hardware={connection.performance.hardware_count if connection.performance else 0} "
            f"ranking={self._format_reported_hardware(connection)}",
            flush=True,
        )
        self._print_cluster_compute_capacity()
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
                status_code=STATUS_OK,
                worker_count=worker_count,
                client_count=client_count,
                client_id=connection.runtime_id,
            ),
        )
        print(
            f"Registered client {connection.node_name} "
            f"id={connection.runtime_id} "
            f"from {connection.peer_address}:{connection.peer_port}",
            flush=True,
        )
        client_thread = threading.Thread(
            target=self._serve_client_connection,
            args=(connection,),
            name=f"superweb-client-{connection.node_name}",
            daemon=True,
        )
        client_thread.start()
        return connection

    @trace_function
    def _register_runtime_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
    ) -> RuntimePeerConnection:
        """Receive and accept one worker or client registration."""

        client_sock.settimeout(self.config.runtime_socket_timeout)
        message = recv_message(client_sock, max_size=self.config.max_message_size)
        if message is None:
            raise ConnectionError("peer closed the TCP session before registration")

        if message.kind == MessageKind.REGISTER_WORKER and message.register_worker is not None:
            return self._register_worker_connection(client_sock, addr, main_node_ip, message.register_worker)
        if message.kind == MessageKind.CLIENT_JOIN and message.client_join is not None:
            return self._register_client_connection(client_sock, addr, message.client_join)

        raise ValueError(
            f"expected {RUNTIME_MSG_REGISTER_WORKER} or {RUNTIME_MSG_CLIENT_JOIN}, got {describe_message_kind(message.kind)}"
        )

    @trace_function
    def _accept_runtime_connections(self, server_sock: socket.socket, main_node_ip: str) -> None:
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
                self._register_runtime_connection(client_sock, addr, main_node_ip)
            except (OSError, ValueError, ConnectionError) as exc:
                print(f"Rejected superweb-cluster runtime connection from {addr[0]}:{addr[1]}: {exc}", flush=True)
                network.safe_close(client_sock)

    @trace_function
    def _remove_worker_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        removed = self.registry.remove_worker(connection.peer_id)
        if removed is not None:
            network.safe_close(removed.sock)
        print(
            f"Removed compute node {connection.node_name} at {connection.peer_address}:{connection.peer_port}: {reason}",
            flush=True,
        )
        self._print_cluster_compute_capacity()

    def _ensure_spatial_convolution_dataset_ready(self) -> None:
        runtime_input = self.spatial_dataset_dir / "runtime_input.bin"
        runtime_weight = self.spatial_dataset_dir / "runtime_weight.bin"
        test_input = self.spatial_dataset_dir / "test_input.bin"
        test_weight = self.spatial_dataset_dir / "test_weight.bin"
        if runtime_input.exists() and runtime_weight.exists() and test_input.exists() and test_weight.exists():
            return

        subprocess.run(
            [
                sys.executable,
                str(SPATIAL_DATASET_GENERATE_SCRIPT_PATH),
                "--output-dir",
                str(self.spatial_dataset_dir),
                "--role",
                "main",
                "--include-runtime-weight",
            ],
            check=True,
            timeout=1800.0,
        )

    def _spatial_weight_slice(self, *, variant: str, spec, start_oc: int, end_oc: int) -> bytes:
        weight_path = self.spatial_dataset_dir / f"{variant}_weight.bin"
        if not weight_path.exists():
            raise FileNotFoundError(f"missing spatial_convolution weight file at {weight_path}")
        weight_data = weight_path.read_bytes()
        bytes_per_output_channel = spec.k * spec.k * spec.c_in * 4
        return weight_data[start_oc * bytes_per_output_channel:end_oc * bytes_per_output_channel]

    def _max_spatial_channels_per_task(self, spec) -> int:
        bytes_per_channel = spec.output_h * spec.output_w * 4
        if bytes_per_channel <= 0:
            raise ValueError("spatial_convolution output channel size must be positive")
        payload_budget = max(self.config.max_message_size - 4096, 1)
        return max(1, payload_budget // bytes_per_channel)

    @trace_function
    def _run_worker_task_slice(self, request, assignment: WorkerTaskSlice):
        exchange_timeout = max(self.config.runtime_socket_timeout, 30.0)
        sock = assignment.connection.sock
        previous_timeout = sock.gettimeout()
        build_kwargs = dict(
            request_id=request.request_id,
            node_id=assignment.connection.runtime_id,
            task_id=assignment.task_id,
            method=request.method,
            object_id=request.object_id,
            stream_id=request.stream_id,
            row_start=assignment.row_start,
            row_end=assignment.row_end,
            vector_data=request.vector_data,
            vector_length=request.vector_length,
            iteration_count=request.iteration_count,
        )
        if request.method == METHOD_SPATIAL_CONVOLUTION:
            spec, variant = load_named_workload_spec(request.object_id)
            build_kwargs.update(
                start_oc=assignment.start_oc,
                end_oc=assignment.end_oc,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=self._spatial_weight_slice(
                    variant=variant,
                    spec=spec,
                    start_oc=assignment.start_oc,
                    end_oc=assignment.end_oc,
                ),
            )
        task_assign = build_task_assign(
            **build_kwargs,
        )

        try:
            with assignment.connection.io_lock:
                sock.settimeout(exchange_timeout)
                send_message(sock, task_assign)
                task_scope = (
                    f"oc={assignment.start_oc}:{assignment.end_oc}"
                    if request.method == METHOD_SPATIAL_CONVOLUTION
                    else f"rows={assignment.row_start}:{assignment.row_end}"
                )
                print(
                    f"{RUNTIME_MSG_TASK_ASSIGN} to {assignment.connection.node_name} "
                    f"id={assignment.connection.runtime_id} "
                    f"task_id={assignment.task_id} method={request.method} {task_scope} "
                    f"iteration_count={request.iteration_count}",
                    flush=True,
                )

                accept_message = recv_message(sock, max_size=self.config.max_message_size)
                if accept_message is None:
                    raise ConnectionError("worker closed the TCP session before TASK_ACCEPT")
                if accept_message.kind != MessageKind.TASK_ACCEPT or accept_message.task_accept is None:
                    raise ValueError(
                        f"expected {RUNTIME_MSG_TASK_ACCEPT}, got {describe_message_kind(accept_message.kind)}"
                    )
                task_accept = accept_message.task_accept
                if (
                    task_accept.request_id != request.request_id
                    or task_accept.task_id != assignment.task_id
                    or task_accept.node_id != assignment.connection.runtime_id
                ):
                    raise ValueError("received TASK_ACCEPT for the wrong request or task id")

                result_message = recv_message(sock, max_size=self.config.max_message_size)
                if result_message is None:
                    raise ConnectionError("worker closed the TCP session before TASK_RESULT")
                if result_message.kind == MessageKind.TASK_FAIL and result_message.task_fail is not None:
                    task_fail = result_message.task_fail
                    if (
                        task_fail.request_id != request.request_id
                        or task_fail.task_id != assignment.task_id
                        or task_fail.node_id != assignment.connection.runtime_id
                    ):
                        raise ValueError("received TASK_FAIL for the wrong request or task id")
                    raise RuntimeError(task_fail.error_message or "worker reported TASK_FAIL")
                if result_message.kind != MessageKind.TASK_RESULT or result_message.task_result is None:
                    raise ValueError(
                        f"expected {RUNTIME_MSG_TASK_RESULT}, got {describe_message_kind(result_message.kind)}"
                    )
                task_result = result_message.task_result
                if (
                    task_result.request_id != request.request_id
                    or task_result.task_id != assignment.task_id
                    or task_result.node_id != assignment.connection.runtime_id
                ):
                    raise ValueError("received TASK_RESULT for the wrong request or task id")
                return task_result
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            self._remove_worker_connection(assignment.connection, str(exc))
            raise
        finally:
            try:
                sock.settimeout(previous_timeout)
            except OSError:
                pass

    @trace_function
    def _build_client_response_for_request(self, request):
        worker_count, client_count = self._cluster_counts()

        if request.method not in (METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION):
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=f"unsupported method: {request.method}",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION and request.vector_length != self.fixed_matvec_spec.cols:
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=(
                    f"vector_length={request.vector_length} does not match "
                    f"expected {self.fixed_matvec_spec.cols}"
                ),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION and len(request.vector_data) != request.vector_length * 4:
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="vector_data byte length does not match vector_length",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )
        if request.iteration_count <= 0:
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="iteration_count must be positive",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=1,
            )

        workers = self.registry.list_workers()
        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
            assignments = self.dispatcher.dispatch_fixed_matrix_vector_multiplication(
                request_id=request.request_id,
                rows=self.fixed_matvec_spec.rows,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION),
            )
        else:
            try:
                spec, variant = load_named_workload_spec(request.object_id)
            except ValueError as exc:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=str(exc),
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            max_channels_per_task = self._max_spatial_channels_per_task(spec)
            if max_channels_per_task <= 0:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message="spatial_convolution workload is too large for in-band task transport",
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            if spec.output_h * spec.output_w * 4 > self.config.max_message_size:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=(
                        "spatial_convolution runtime output exceeds the current per-task message limit; "
                        "artifact side-channel support is still pending"
                    ),
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            try:
                self._ensure_spatial_convolution_dataset_ready()
            except (OSError, subprocess.CalledProcessError) as exc:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_INTERNAL_ERROR,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=f"failed to prepare spatial_convolution dataset: {exc}",
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            assignments = self.dispatcher.dispatch_spatial_convolution(
                request_id=request.request_id,
                output_channels=spec.c_out,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_SPATIAL_CONVOLUTION),
                max_channels_per_task=max_channels_per_task,
            )
        if not assignments:
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_NOT_FOUND,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="no registered compute workers are currently available",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        try:
            with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="task-dispatch") as executor:
                task_results = list(executor.map(lambda item: self._run_worker_task_slice(request, item), assignments))
            if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
                output_vector = self.aggregator.collect_fixed_matrix_vector_result(
                    rows=self.fixed_matvec_spec.rows,
                    results=task_results,
                )
                output_length = self.fixed_matvec_spec.rows
                result_artifact_id = ""
            else:
                spec, _variant = load_named_workload_spec(request.object_id)
                output_vector = self.aggregator.collect_spatial_convolution_result(
                    out_h=spec.output_h,
                    out_w=spec.output_w,
                    total_cout=spec.c_out,
                    results=task_results,
                )
                output_length = spec.output_h * spec.output_w * spec.c_out
                result_artifact_id = ""
                if len(output_vector) > self.config.max_message_size:
                    artifact_dir = self.spatial_dataset_dir / "artifacts"
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    artifact_path = artifact_dir / f"{request.request_id}.bin"
                    artifact_path.write_bytes(output_vector)
                    output_vector = b""
                    result_artifact_id = str(artifact_path)
            worker_count, client_count = self._cluster_counts()
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_OK,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                output_vector=output_vector,
                output_length=output_length,
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
                result_artifact_id=result_artifact_id,
            )
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            worker_count, client_count = self._cluster_counts()
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_INTERNAL_ERROR,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=str(exc),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

    @trace_function
    def _remove_client_connection(self, connection: RuntimePeerConnection, reason: str) -> None:
        removed = self.registry.remove_client(connection.peer_id)
        if removed is not None:
            network.safe_close(removed.sock)
        print(
            f"Removed client {connection.node_name} at {connection.peer_address}:{connection.peer_port}: {reason}",
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
                f"request_id={request.request_id} method={request.method} "
                f"vector_length={request.vector_length} iteration_count={request.iteration_count}",
                flush=True,
            )
            response = self._build_client_response_for_request(request)
            if response.client_response is not None:
                response.client_response.client_id = connection.runtime_id
            send_message(connection.sock, response)
            print(
                f"{RUNTIME_MSG_CLIENT_RESPONSE} to {request.client_name} "
                f"request_id={request.request_id} status_code={response.client_response.status_code}",
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
            heartbeat = build_heartbeat(MAIN_NODE_NAME)
            assert heartbeat.heartbeat is not None
            heartbeat_unix_time_ms = heartbeat.heartbeat.unix_time_ms

            try:
                with connection.io_lock:
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

        self._remove_worker_connection(connection, f"after heartbeat timeout: {last_error}")

    @trace_function
    def _send_heartbeat_once(self) -> None:
        """Send one main-node heartbeat cycle to all registered workers."""

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
        """Reply to main-node browse queries."""

        if not multicast.parse_discover_message(message):
            return

        description = multicast.describe_packet(message)
        print(f"mDNS packet from {addr[0]}:{addr[1]} -> {description}", flush=True)
        announce_host = multicast.send_announce(endpoint, addr, self.config, MAIN_NODE_NAME)
        print(f"Sent main-node mDNS response using {announce_host}:{self.config.tcp_port}", flush=True)

    @trace_function
    def run(self) -> DiscoveryResult:
        """Run the main-node multicast loop until shutdown is requested."""

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
            network.set_socket_timeout(endpoint.sock, self.config.main_node_poll_timeout)

            local_ip = network.resolve_local_ip()
            local_mac = network.get_local_mac_address()
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
            self._close_runtime_connections()
            multicast.close(endpoint)


