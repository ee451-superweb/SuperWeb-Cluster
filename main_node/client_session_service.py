"""Own the runtime request loop for one registered client session.

Use this module when the main node needs to read ``CLIENT_REQUEST`` messages,
track client request state in the registry, and send responses back over the
runtime TCP session.
"""

from __future__ import annotations

import socket
import time
from contextlib import nullcontext

from adapters.audit_log import write_audit_event
from core.constants import METHOD_CONV2D, RUNTIME_MSG_CLIENT_REQUEST, RUNTIME_MSG_CLIENT_RESPONSE, STATUS_OK
from core.tracing import trace_function
from compute_node.compute_methods.conv2d.executor import load_named_workload_spec
from main_node.mailbox import RuntimeConnectionMailbox
from wire.internal_protocol.transport import (
    MessageKind,
    build_client_request_ok,
    describe_message_kind,
    recv_message,
    send_message,
)


def _resolve_connection_lock(lock):
    """Return a usable context manager for connection I/O locks.

    Args:
        lock: Candidate lock-like object stored on a runtime connection.

    Returns:
        The original lock when it supports ``with``, otherwise ``nullcontext()``.
    """
    lock_type = type(lock)
    if callable(getattr(lock_type, "__enter__", None)) and callable(getattr(lock_type, "__exit__", None)):
        return lock
    return nullcontext()


class ClientSessionService:
    """Serve one registered client from join until disconnect."""

    def __init__(
        self,
        *,
        config,
        registry,
        build_client_response_for_request,
        allocate_data_plane_endpoints,
        remove_client_connection,
        logger=None,
    ) -> None:
        """Store runtime dependencies needed to serve a client session.

        Args:
            config: Main-node runtime configuration with message-size limits.
            registry: Registry used to track per-client request state.
            build_client_response_for_request: Handler that executes one request
                given a pre-registered ``DataPlaneAllocation``.
            allocate_data_plane_endpoints: Callable that pre-registers upload
                and download IDs on the artifact manager before REQUEST_OK.
            remove_client_connection: Callback that removes broken client sockets.
            logger: Logger used for audit events and unexpected-message warnings.
        """
        self.config = config
        self.registry = registry
        self.build_client_response_for_request = build_client_response_for_request
        self.allocate_data_plane_endpoints = allocate_data_plane_endpoints
        self.remove_client_connection = remove_client_connection
        self.logger = logger

    def describe_client_request(self, request) -> str:
        """Render a concise workload summary for client-request logs.

        Use this when printing incoming client requests so operators can see the
        major dimensions of the request without inspecting raw payload bytes.

        Args:
            request: Parsed client request dataclass.

        Returns:
            A short textual summary of the workload dimensions.
        """
        if request.method == METHOD_CONV2D:
            if request.object_id:
                try:
                    spec, variant = load_named_workload_spec(request.object_id, size=request.size)
                except ValueError:
                    pass
                else:
                    return (
                        f"size={variant} tensor={spec.h}x{spec.w} "
                        f"channels={spec.c_in}->{spec.c_out} "
                        f"kernel={spec.k} pad={spec.pad} stride={spec.stride}"
                    )
            return (
                f"size={request.size or '<unspecified>'} "
                f"tensor={request.tensor_h}x{request.tensor_w} "
                f"channels={request.channels_in}->{request.channels_out} "
                f"kernel={request.kernel_size} pad={request.padding} stride={request.stride}"
            )
        return f"size={request.size or '<unspecified>'} vector_length={request.vector_length}"

    def describe_client_response(self, response) -> str:
        """Render result, artifact, and latency details for client-response logs.

        Args:
            response: Runtime envelope returned by the request handler.

        Returns:
            A short suffix describing elapsed time and result transport details.
        """
        payload = response.client_response
        if payload is None:
            return ""
        parts = [f"task_id={payload.task_id or '<unassigned>'}"]
        if payload.status_code != STATUS_OK and payload.error_message:
            parts.append(f"error={payload.error_message!r}")
        if payload.elapsed_ms > 0:
            parts.append(f"elapsed_ms={payload.elapsed_ms}")
        if payload.timing is not None:
            parts.append(f"dispatch_ms={payload.timing.dispatch_ms}")
            parts.append(f"task_window_ms={payload.timing.task_window_ms}")
            parts.append(f"aggregate_ms={payload.timing.aggregate_ms}")
            for worker in payload.timing.workers:
                suffix = (
                    (f"+fetch:{worker.artifact_fetch_ms}ms" if worker.artifact_fetch_ms else "")
                    + (f"+compute:{worker.computation_ms}ms" if worker.computation_ms else "")
                    + (f"+peripheral:{worker.peripheral_ms}ms" if worker.peripheral_ms else "")
                )
                parts.append(
                    f"worker[{worker.node_id}|{worker.slice}]"
                    f"=wall:{worker.wall_ms}ms{suffix}"
                )
        if payload.result_artifact is not None:
            parts.append(f"artifact_id={payload.result_artifact.artifact_id}")
            parts.append(f"artifact_bytes={payload.result_artifact.size_bytes}")
        elif payload.output_vector:
            parts.append(f"inline_bytes={len(payload.output_vector)}")
        if payload.output_length:
            parts.append(f"output_length={payload.output_length}")
        return " " + " ".join(parts)

    def summarize_client_response(self, client_name: str, response) -> str:
        """Render the stdout-safe client-response summary for operators.

        Keep terminal output compact: show only the protocol name, client name,
        status code, and task id. The detailed response shape stays in file
        logs through ``describe_client_response``.
        """

        payload = response.client_response
        if payload is None:
            return f"{RUNTIME_MSG_CLIENT_RESPONSE} to {client_name}"
        return (
            f"{RUNTIME_MSG_CLIENT_RESPONSE} to {client_name} "
            f"status_code={payload.status_code} "
            f"task_id={payload.task_id or '<unassigned>'}"
        )

    @trace_function
    def serve_client_connection(self, connection, runtime_should_stop) -> None:
        """Run the request/response loop for one registered client.

        Use this in the background client thread created after ``CLIENT_JOIN`` so
        the main node can continuously handle requests on that socket.

        Args:
            connection: Registered client connection object.
            runtime_should_stop: Callback that reports shutdown state.

        Returns:
            ``None`` after the client disconnects or is removed.
        """
        mailbox = getattr(connection, "mailbox", None)
        while not runtime_should_stop():
            try:
                if isinstance(mailbox, RuntimeConnectionMailbox):
                    message = mailbox.wait_for_client_request(self.config.runtime_socket_timeout)
                else:
                    message = recv_message(connection.sock, max_size=self.config.max_message_size)
            except socket.timeout:
                continue
            except (OSError, ValueError, ConnectionError) as exc:
                self.remove_client_connection(connection, str(exc))
                return

            if message is None:
                self.remove_client_connection(connection, "client closed the TCP session")
                return

            if message.kind != MessageKind.CLIENT_REQUEST or message.client_request is None:
                if self.logger is not None:
                    self.logger.warning(
                        "Ignoring unexpected client runtime message from %s: %s",
                        connection.node_name,
                        describe_message_kind(message.kind),
                    )
                continue

            request = message.client_request
            self.registry.mark_client_request(connection.peer_id)
            task_id = self.registry.allocate_task_id(request.method)
            request.request_id = task_id
            if hasattr(self.registry, "mark_client_request_state"):
                self.registry.mark_client_request_state(
                    connection.peer_id,
                    task_id=task_id,
                    method=request.method,
                )
            write_audit_event(
                f"{RUNTIME_MSG_CLIENT_REQUEST} from {request.client_name} "
                f"task_id={task_id} method={request.method} "
                f"{self.describe_client_request(request)} iteration_count={request.iteration_count}",
                stdout=True,
                logger=self.logger,
            )
            try:
                accepted_timestamp_ms = int(time.time() * 1000)
                allocation = self.allocate_data_plane_endpoints(request)
                with _resolve_connection_lock(getattr(connection, "io_lock", None)):
                    send_message(
                        connection.sock,
                        build_client_request_ok(
                            client_id=connection.runtime_id,
                            task_id=task_id,
                            method=request.method,
                            size=request.size,
                            object_id=request.object_id,
                            accepted_timestamp_ms=accepted_timestamp_ms,
                            upload_id=allocation.upload_id,
                            download_id=allocation.download_id,
                            data_endpoint_host=allocation.data_endpoint_host,
                            data_endpoint_port=allocation.data_endpoint_port,
                        ),
                    )
                response = self.build_client_response_for_request(
                    request,
                    allocation=allocation,
                )
                if response.client_response is not None:
                    response.client_response.client_id = connection.runtime_id
                    if not response.client_response.task_id:
                        response.client_response.task_id = task_id
                with _resolve_connection_lock(getattr(connection, "io_lock", None)):
                    send_message(connection.sock, response)
                write_audit_event(
                    f"{RUNTIME_MSG_CLIENT_RESPONSE} to {request.client_name} "
                    f"status_code={response.client_response.status_code}"
                    f"{self.describe_client_response(response)}",
                    logger=self.logger,
                )
                print(self.summarize_client_response(request.client_name, response), flush=True)
            except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
                self.remove_client_connection(connection, str(exc))
                return
            finally:
                if hasattr(self.registry, "clear_client_request_state"):
                    self.registry.clear_client_request_state(
                        connection.peer_id,
                        task_id=task_id,
                    )
