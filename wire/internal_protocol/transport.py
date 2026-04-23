"""Control-plane transport and builder helpers for runtime envelopes.

Use this module when Python code needs framing, send/receive helpers, and
builder functions for workers, clients, heartbeats, tasks, and artifacts. The
underlying message models live in the dedicated internal/external control-plane
and data-plane model modules.
"""

from __future__ import annotations

import socket
import struct
import time

from core.constants import (
    MAIN_NODE_NAME,
    RUNTIME_MSG_ARTIFACT_RELEASE,
    RUNTIME_MSG_CLIENT_INFO_REPLY,
    RUNTIME_MSG_CLIENT_INFO_REQUEST,
    RUNTIME_MSG_CLIENT_JOIN,
    RUNTIME_MSG_CLIENT_REQUEST,
    RUNTIME_MSG_CLIENT_REQUEST_OK,
    RUNTIME_MSG_CLIENT_RESPONSE,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_REGISTER_OK,
    RUNTIME_MSG_REGISTER_WORKER,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
    RUNTIME_MSG_WORKER_UPDATE,
    SUPERWEB_CLIENT_NAME,
)
from core.tracing import trace_function
from core.types import ComputePerformanceSummary, HardwareProfile
from wire.external_protocol.control_plane import (
    ClientInfoReply,
    ClientInfoRequest,
    ClientJoin,
    ClientRequest,
    ClientRequestOk,
    ClientResponse,
    GemmRequestPayload,
    GemmResponsePayload,
    GemvRequestPayload,
    GemvResponsePayload,
    Conv2dRequestPayload,
    Conv2dResponsePayload,
    ResponseTiming,
    WorkerTiming,
)
from wire.external_protocol.data_plane import ArtifactDescriptor
from wire.internal_protocol.common import MessageKind, NodeStatus, RuntimeEnvelope, TransferMode
from wire.internal_protocol.control_plane import (
    ArtifactRelease,
    GemmResultPayload,
    GemmTaskPayload,
    GemvResultPayload,
    GemvTaskPayload,
    Heartbeat,
    HeartbeatOk,
    RegisterOk,
    RegisterWorker,
    Conv2dResultPayload,
    Conv2dTaskPayload,
    TaskAccept,
    TaskAssign,
    TaskFail,
    TaskResult,
    WorkerUpdate,
)
from wire.internal_protocol.control_plane_codec import encode_envelope, parse_envelope

FRAME_HEADER = struct.Struct("!I")


def _recv_exactly(sock: socket.socket, size: int) -> bytes | None:
    """Use this internal helper when one framed runtime read needs exact bytes.

    Args:
        sock: Connected TCP socket to read from.
        size: Exact byte count required for the next frame segment.

    Returns:
        The requested bytes, ``None`` on clean EOF before any bytes arrive, or
        raises on a mid-frame close.
    """
    remaining = size
    chunks: list[bytes] = []

    while remaining:
        chunk = sock.recv(remaining)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(
                "socket.recv() must return bytes-like data, "
                f"got {type(chunk).__name__}"
            )
        chunk = bytes(chunk)
        if not chunk:
            if chunks:
                raise ConnectionError("peer closed the TCP session mid-frame")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


@trace_function
def send_message(sock: socket.socket, message: RuntimeEnvelope) -> None:
    """Use this when one runtime envelope should be sent on a connected socket."""
    payload = encode_envelope(message)
    sock.sendall(FRAME_HEADER.pack(len(payload)))
    sock.sendall(payload)


def recv_message(sock: socket.socket, *, max_size: int) -> RuntimeEnvelope | None:
    """Use this when one runtime socket should yield the next framed protobuf envelope."""
    header = _recv_exactly(sock, FRAME_HEADER.size)
    if header is None:
        return None

    (payload_size,) = FRAME_HEADER.unpack(header)
    if payload_size <= 0 or payload_size > max_size:
        raise ValueError(f"invalid protobuf message size: {payload_size}")

    payload = _recv_exactly(sock, payload_size)
    if payload is None:
        return None

    return parse_envelope(payload)


@trace_function
def build_register_worker(
    node_name: str,
    hardware: HardwareProfile,
    performance: ComputePerformanceSummary | None = None,
) -> RuntimeEnvelope:
    """Use this when a compute node first registers with the main node."""
    if performance is None:
        performance = ComputePerformanceSummary()

    return RuntimeEnvelope(
        kind=MessageKind.REGISTER_WORKER,
        register_worker=RegisterWorker(node_name=node_name, hardware=hardware, performance=performance),
    )


@trace_function
def build_register_ok(
    main_node_ip: str,
    main_node_port: int,
    *,
    node_id: str = "",
    main_node_name: str = MAIN_NODE_NAME,
) -> RuntimeEnvelope:
    """Use this when the main node accepts a worker registration."""
    return RuntimeEnvelope(
        kind=MessageKind.REGISTER_OK,
        register_ok=RegisterOk(
            main_node_name=main_node_name,
            main_node_ip=main_node_ip,
            main_node_port=main_node_port,
            node_id=node_id,
        ),
    )


@trace_function
def build_heartbeat(main_node_name: str = MAIN_NODE_NAME, unix_time_ms: int | None = None) -> RuntimeEnvelope:
    """Use this when the main node needs to poll one worker or client for liveness."""
    if unix_time_ms is None:
        unix_time_ms = int(time.time() * 1000)

    return RuntimeEnvelope(
        kind=MessageKind.HEARTBEAT,
        heartbeat=Heartbeat(main_node_name=main_node_name, unix_time_ms=unix_time_ms),
    )


@trace_function
def build_heartbeat_ok(
    node_name: str,
    heartbeat_unix_time_ms: int,
    received_unix_time_ms: int | None = None,
    *,
    node_id: str = "",
    active_task_ids: list[str] | tuple[str, ...] | None = None,
    node_status: NodeStatus = NodeStatus.UNKNOWN,
    completed_task_count: int = 0,
) -> RuntimeEnvelope:
    """Use this when a worker or client replies to one heartbeat probe."""
    if received_unix_time_ms is None:
        received_unix_time_ms = int(time.time() * 1000)
    if active_task_ids is None:
        active_task_ids = ()

    return RuntimeEnvelope(
        kind=MessageKind.HEARTBEAT_OK,
        heartbeat_ok=HeartbeatOk(
            node_name=node_name,
            heartbeat_unix_time_ms=heartbeat_unix_time_ms,
            received_unix_time_ms=received_unix_time_ms,
            active_task_ids=tuple(active_task_ids),
            node_status=node_status,
            completed_task_count=completed_task_count,
            node_id=node_id,
        ),
    )


@trace_function
def build_client_join(client_name: str = SUPERWEB_CLIENT_NAME) -> RuntimeEnvelope:
    """Use this when a client opens a new runtime session with the main node."""
    return RuntimeEnvelope(kind=MessageKind.CLIENT_JOIN, client_join=ClientJoin(client_name=client_name))


@trace_function
def build_client_info_request(
    *,
    client_id: str,
    client_name: str,
    timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when a joined client polls the main node for liveness and task state."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.CLIENT_INFO_REQUEST,
        client_info_request=ClientInfoRequest(
            client_id=client_id,
            client_name=client_name,
            timestamp_ms=timestamp_ms,
        ),
    )


@trace_function
def build_client_info_reply(
    *,
    client_id: str,
    request_timestamp_ms: int,
    timeout_ms: int,
    has_active_tasks: bool,
    active_task_ids: list[str] | tuple[str, ...] | None = None,
    reply_timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when the main node answers one client liveness/info poll."""
    if reply_timestamp_ms is None:
        reply_timestamp_ms = int(time.time() * 1000)
    if active_task_ids is None:
        active_task_ids = ()
    return RuntimeEnvelope(
        kind=MessageKind.CLIENT_INFO_REPLY,
        client_info_reply=ClientInfoReply(
            client_id=client_id,
            request_timestamp_ms=request_timestamp_ms,
            reply_timestamp_ms=reply_timestamp_ms,
            timeout_ms=timeout_ms,
            has_active_tasks=has_active_tasks,
            active_task_ids=tuple(active_task_ids),
        ),
    )


@trace_function
def build_client_request(
    client_name: str,
    request_id: str,
    method: str,
    vector_data: bytes,
    *,
    size: str = "",
    object_id: str = "",
    stream_id: str = "",
    timestamp_ms: int | None = None,
    vector_length: int | None = None,
    iteration_count: int = 1,
    tensor_h: int = 0,
    tensor_w: int = 0,
    channels_in: int = 0,
    channels_out: int = 0,
    kernel_size: int = 0,
    padding: int = 0,
    stride: int = 1,
    conv2d_client_response_mode: int = 0,
    conv2d_stats_max_samples: int = 0,
    request_payload: GemvRequestPayload | Conv2dRequestPayload | GemmRequestPayload | None = None,
) -> RuntimeEnvelope:
    """Use this when a client submits one structured compute request."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    if vector_length is None:
        vector_length = len(vector_data) // 4
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")

    return RuntimeEnvelope(
        kind=MessageKind.CLIENT_REQUEST,
        client_request=ClientRequest(
            request_id=request_id,
            client_name=client_name,
            method=method,
            size=size,
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            iteration_count=iteration_count,
            request_payload=request_payload,
            vector_length=vector_length,
            vector_data=vector_data,
            tensor_h=tensor_h,
            tensor_w=tensor_w,
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            conv2d_client_response_mode=conv2d_client_response_mode,
            conv2d_stats_max_samples=conv2d_stats_max_samples,
        ),
    )


@trace_function
def build_client_request_ok(
    *,
    client_id: str,
    task_id: str,
    method: str,
    size: str = "",
    object_id: str,
    accepted_timestamp_ms: int | None = None,
    upload_id: str = "",
    download_id: str = "",
    data_endpoint_host: str = "",
    data_endpoint_port: int = 0,
) -> RuntimeEnvelope:
    """Use this when the main node assigns a task id for one client request."""
    if accepted_timestamp_ms is None:
        accepted_timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.CLIENT_REQUEST_OK,
        client_request_ok=ClientRequestOk(
            client_id=client_id,
            task_id=task_id,
            method=method,
            size=size,
            object_id=object_id,
            accepted_timestamp_ms=accepted_timestamp_ms,
            upload_id=upload_id,
            download_id=download_id,
            data_endpoint_host=data_endpoint_host,
            data_endpoint_port=data_endpoint_port,
        ),
    )


@trace_function
def build_client_response(
    request_id: str,
    status_code: int,
    *,
    method: str = "",
    size: str = "",
    object_id: str = "",
    stream_id: str = "",
    error_message: str = "",
    output_vector: bytes = b"",
    timestamp_ms: int | None = None,
    output_length: int | None = None,
    worker_count: int = 0,
    client_count: int = 0,
    client_id: str = "",
    iteration_count: int = 1,
    task_id: str = "",
    elapsed_ms: int = 0,
    result_artifact_id: str = "",
    result_artifact: ArtifactDescriptor | None = None,
    response_payload: GemvResponsePayload | Conv2dResponsePayload | GemmResponsePayload | None = None,
    timing: ResponseTiming | None = None,
) -> RuntimeEnvelope:
    """Use this when the main node replies to CLIENT_JOIN or CLIENT_REQUEST."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    if output_length is None:
        output_length = len(output_vector) // 4
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")

    return RuntimeEnvelope(
        kind=MessageKind.CLIENT_RESPONSE,
        client_response=ClientResponse(
            request_id=request_id,
            method=method,
            size=size,
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            status_code=status_code,
            error_message=error_message,
            worker_count=worker_count,
            client_count=client_count,
            client_id=client_id,
            iteration_count=iteration_count,
            task_id=task_id,
            elapsed_ms=elapsed_ms,
            response_payload=response_payload,
            result_artifact=result_artifact,
            result_artifact_id=result_artifact_id,
            output_length=output_length,
            output_vector=output_vector,
            timing=timing,
        ),
    )


@trace_function
def build_task_assign(
    request_id: str,
    node_id: str,
    task_id: str,
    method: str,
    size: str = "",
    row_start: int = 0,
    row_end: int = 0,
    vector_data: bytes = b"",
    *,
    object_id: str = "",
    stream_id: str = "",
    timestamp_ms: int | None = None,
    vector_length: int | None = None,
    iteration_count: int = 1,
    transfer_mode: TransferMode = TransferMode.UNSPECIFIED,
    artifact_id: str = "",
    artifact_timeout_ms: int = 0,
    start_oc: int = 0,
    end_oc: int = 0,
    tensor_h: int = 0,
    tensor_w: int = 0,
    channels_in: int = 0,
    channels_out: int = 0,
    kernel_size: int = 0,
    padding: int = 0,
    stride: int = 1,
    weight_data: bytes = b"",
    m_start: int = 0,
    m_end: int = 0,
    m: int = 0,
    n: int = 0,
    k: int = 0,
    task_payload: GemvTaskPayload | Conv2dTaskPayload | GemmTaskPayload | None = None,
) -> RuntimeEnvelope:
    """Use this when the main node dispatches one worker slice of a client request."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    if vector_length is None:
        vector_length = len(vector_data) // 4
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")

    return RuntimeEnvelope(
        kind=MessageKind.TASK_ASSIGN,
        task_assign=TaskAssign(
            request_id=request_id,
            node_id=node_id,
            task_id=task_id,
            method=method,
            size=size,
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            iteration_count=iteration_count,
            transfer_mode=transfer_mode,
            artifact_id=artifact_id,
            artifact_timeout_ms=artifact_timeout_ms,
            task_payload=task_payload,
            row_start=row_start,
            row_end=row_end,
            vector_length=vector_length,
            vector_data=vector_data,
            start_oc=start_oc,
            end_oc=end_oc,
            tensor_h=tensor_h,
            tensor_w=tensor_w,
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            weight_data=weight_data,
            m_start=m_start,
            m_end=m_end,
            m=m,
            n=n,
            k=k,
        ),
    )


@trace_function
def build_task_accept(
    request_id: str,
    node_id: str,
    task_id: str,
    status_code: int,
    *,
    timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when a worker accepts a TASK_ASSIGN message."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.TASK_ACCEPT,
        task_accept=TaskAccept(
            request_id=request_id,
            node_id=node_id,
            task_id=task_id,
            timestamp_ms=timestamp_ms,
            status_code=status_code,
        ),
    )


@trace_function
def build_task_fail(
    request_id: str,
    node_id: str,
    task_id: str,
    status_code: int,
    error_message: str,
    *,
    timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when a worker cannot complete an assigned task."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.TASK_FAIL,
        task_fail=TaskFail(
            request_id=request_id,
            node_id=node_id,
            task_id=task_id,
            timestamp_ms=timestamp_ms,
            status_code=status_code,
            error_message=error_message,
        ),
    )


@trace_function
def build_task_result(
    request_id: str,
    node_id: str,
    task_id: str,
    status_code: int,
    row_start: int = 0,
    row_end: int = 0,
    output_vector: bytes = b"",
    *,
    timestamp_ms: int | None = None,
    output_length: int | None = None,
    iteration_count: int = 1,
    start_oc: int = 0,
    end_oc: int = 0,
    output_h: int = 0,
    output_w: int = 0,
    result_artifact_id: str = "",
    result_artifact: ArtifactDescriptor | None = None,
    result_payload: GemvResultPayload | Conv2dResultPayload | GemmResultPayload | None = None,
    computation_ms: int = 0,
    peripheral_ms: int = 0,
    m_start: int = 0,
    m_end: int = 0,
    method: str = "",
) -> RuntimeEnvelope:
    """Use this when a worker completes one assigned task slice."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    if output_length is None:
        output_length = len(output_vector) // 4
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")

    return RuntimeEnvelope(
        kind=MessageKind.TASK_RESULT,
        task_result=TaskResult(
            request_id=request_id,
            node_id=node_id,
            task_id=task_id,
            timestamp_ms=timestamp_ms,
            status_code=status_code,
            iteration_count=iteration_count,
            result_payload=result_payload,
            result_artifact=result_artifact,
            row_start=row_start,
            row_end=row_end,
            output_length=output_length,
            output_vector=output_vector,
            start_oc=start_oc,
            end_oc=end_oc,
            output_h=output_h,
            output_w=output_w,
            result_artifact_id=result_artifact_id,
            computation_ms=computation_ms,
            peripheral_ms=peripheral_ms,
            m_start=m_start,
            m_end=m_end,
            method=method,
        ),
    )


@trace_function
def build_artifact_release(
    *,
    node_id: str,
    task_id: str,
    artifact_id: str,
    timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when the main node is done with a worker-hosted artifact copy."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.ARTIFACT_RELEASE,
        artifact_release=ArtifactRelease(
            node_id=node_id,
            task_id=task_id,
            artifact_id=artifact_id,
            timestamp_ms=timestamp_ms,
        ),
    )


@trace_function
def build_worker_update(
    *,
    node_id: str,
    performance: ComputePerformanceSummary,
    timestamp_ms: int | None = None,
) -> RuntimeEnvelope:
    """Use this when a worker reports updated effective performance data."""
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return RuntimeEnvelope(
        kind=MessageKind.WORKER_UPDATE,
        worker_update=WorkerUpdate(
            node_id=node_id,
            timestamp_ms=timestamp_ms,
            performance=performance,
        ),
    )


def describe_message_kind(kind: MessageKind) -> str:
    """Use this for logs that should display stable human-readable message labels."""
    if kind == MessageKind.REGISTER_WORKER:
        return RUNTIME_MSG_REGISTER_WORKER
    if kind == MessageKind.REGISTER_OK:
        return RUNTIME_MSG_REGISTER_OK
    if kind == MessageKind.HEARTBEAT:
        return RUNTIME_MSG_HEARTBEAT
    if kind == MessageKind.HEARTBEAT_OK:
        return RUNTIME_MSG_HEARTBEAT_OK
    if kind == MessageKind.CLIENT_JOIN:
        return RUNTIME_MSG_CLIENT_JOIN
    if kind == MessageKind.CLIENT_INFO_REQUEST:
        return RUNTIME_MSG_CLIENT_INFO_REQUEST
    if kind == MessageKind.CLIENT_INFO_REPLY:
        return RUNTIME_MSG_CLIENT_INFO_REPLY
    if kind == MessageKind.CLIENT_REQUEST:
        return RUNTIME_MSG_CLIENT_REQUEST
    if kind == MessageKind.CLIENT_REQUEST_OK:
        return RUNTIME_MSG_CLIENT_REQUEST_OK
    if kind == MessageKind.CLIENT_RESPONSE:
        return RUNTIME_MSG_CLIENT_RESPONSE
    if kind == MessageKind.TASK_ASSIGN:
        return RUNTIME_MSG_TASK_ASSIGN
    if kind == MessageKind.TASK_ACCEPT:
        return RUNTIME_MSG_TASK_ACCEPT
    if kind == MessageKind.TASK_FAIL:
        return RUNTIME_MSG_TASK_FAIL
    if kind == MessageKind.TASK_RESULT:
        return RUNTIME_MSG_TASK_RESULT
    if kind == MessageKind.ARTIFACT_RELEASE:
        return RUNTIME_MSG_ARTIFACT_RELEASE
    if kind == MessageKind.WORKER_UPDATE:
        return RUNTIME_MSG_WORKER_UPDATE
    return kind.name


__all__ = [
    "ArtifactDescriptor",
    "ArtifactRelease",
    "ClientInfoReply",
    "ClientInfoRequest",
    "ClientJoin",
    "ClientRequest",
    "ClientRequestOk",
    "ClientResponse",
    "GemmRequestPayload",
    "GemmResponsePayload",
    "GemmResultPayload",
    "GemmTaskPayload",
    "GemvRequestPayload",
    "GemvResponsePayload",
    "GemvResultPayload",
    "GemvTaskPayload",
    "Heartbeat",
    "HeartbeatOk",
    "MessageKind",
    "NodeStatus",
    "RegisterOk",
    "RegisterWorker",
    "ResponseTiming",
    "RuntimeEnvelope",
    "Conv2dRequestPayload",
    "Conv2dResponsePayload",
    "Conv2dResultPayload",
    "Conv2dTaskPayload",
    "TaskAccept",
    "TaskAssign",
    "TaskFail",
    "TaskResult",
    "TransferMode",
    "WorkerTiming",
    "WorkerUpdate",
    "build_artifact_release",
    "build_client_info_reply",
    "build_client_info_request",
    "build_client_join",
    "build_client_request",
    "build_client_request_ok",
    "build_client_response",
    "build_heartbeat",
    "build_heartbeat_ok",
    "build_register_ok",
    "build_register_worker",
    "build_task_accept",
    "build_task_assign",
    "build_task_fail",
    "build_task_result",
    "build_worker_update",
    "describe_message_kind",
    "recv_message",
    "send_message",
]
