"""Minimal protobuf wire-format helpers for the superweb-cluster TCP runtime."""

from __future__ import annotations

import enum
import socket
import struct
import time
from dataclasses import dataclass

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile, MethodPerformanceSummary
from app.constants import (
    MAIN_NODE_NAME,
    RUNTIME_MSG_CLIENT_JOIN,
    RUNTIME_MSG_CLIENT_REQUEST,
    RUNTIME_MSG_CLIENT_RESPONSE,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_REGISTER_OK,
    RUNTIME_MSG_REGISTER_WORKER,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
    SUPERWEB_CLIENT_NAME,
)
from app.trace_utils import trace_function

FRAME_HEADER = struct.Struct("!I")
DOUBLE_LE = struct.Struct("<d")
WIRE_64BIT = 1
WIRE_VARINT = 0
WIRE_LENGTH_DELIMITED = 2


class MessageKind(enum.IntEnum):
    """Supported protobuf envelope types."""

    UNSPECIFIED = 0
    REGISTER_WORKER = 1
    REGISTER_OK = 2
    HEARTBEAT = 3
    HEARTBEAT_OK = 4
    CLIENT_JOIN = 5
    CLIENT_REQUEST = 6
    CLIENT_RESPONSE = 7
    TASK_ASSIGN = 8
    TASK_ACCEPT = 9
    TASK_FAIL = 10
    TASK_RESULT = 11


@dataclass(slots=True)
class RegisterWorker:
    """Compute-node registration payload."""

    node_name: str
    hardware: HardwareProfile
    performance: ComputePerformanceSummary


@dataclass(slots=True)
class RegisterOk:
    """Main-node acceptance payload."""

    main_node_name: str
    main_node_ip: str
    main_node_port: int
    node_id: str


@dataclass(slots=True)
class Heartbeat:
    """One-way main-node heartbeat payload."""

    main_node_name: str
    unix_time_ms: int


@dataclass(slots=True)
class HeartbeatOk:
    """Worker acknowledgement for one heartbeat payload."""

    node_name: str
    heartbeat_unix_time_ms: int
    received_unix_time_ms: int


@dataclass(slots=True)
class ClientJoin:
    """Client registration payload."""

    client_name: str


@dataclass(slots=True)
class ClientRequest:
    """Client request sent to the main node."""

    request_id: str
    client_name: str
    method: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    vector_length: int
    vector_data: bytes
    iteration_count: int


@dataclass(slots=True)
class ClientResponse:
    """Main-node response sent back to a client."""

    request_id: str
    method: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    status_code: int
    error_message: str
    output_length: int
    output_vector: bytes
    worker_count: int
    client_count: int
    client_id: str
    iteration_count: int
    result_artifact_id: str = ""


@dataclass(slots=True)
class RuntimeEnvelope:
    """Framed protobuf message used on the TCP runtime channel."""

    kind: MessageKind
    register_worker: RegisterWorker | None = None
    register_ok: RegisterOk | None = None
    heartbeat: Heartbeat | None = None
    heartbeat_ok: HeartbeatOk | None = None
    client_join: ClientJoin | None = None
    client_request: ClientRequest | None = None
    client_response: ClientResponse | None = None
    task_assign: "TaskAssign" | None = None
    task_accept: "TaskAccept" | None = None
    task_fail: "TaskFail" | None = None
    task_result: "TaskResult" | None = None


@dataclass(slots=True)
class TaskAssign:
    """Main-node instruction telling one worker to compute a row slice."""

    request_id: str
    node_id: str
    task_id: str
    method: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    row_start: int
    row_end: int
    vector_length: int
    vector_data: bytes
    iteration_count: int
    start_oc: int = 0
    end_oc: int = 0
    tensor_h: int = 0
    tensor_w: int = 0
    channels_in: int = 0
    channels_out: int = 0
    kernel_size: int = 0
    padding: int = 0
    stride: int = 1
    weight_data: bytes = b""


@dataclass(slots=True)
class TaskAccept:
    """Worker acknowledgement that one assigned task was accepted."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int


@dataclass(slots=True)
class TaskFail:
    """Worker error result for one assigned task."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int
    error_message: str


@dataclass(slots=True)
class TaskResult:
    """Worker-computed output rows for one assigned task."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int
    row_start: int
    row_end: int
    output_length: int
    output_vector: bytes
    iteration_count: int
    start_oc: int = 0
    end_oc: int = 0
    output_h: int = 0
    output_w: int = 0
    result_artifact_id: str = ""


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("protobuf varints must be non-negative")

    encoded = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            encoded.append(byte | 0x80)
        else:
            encoded.append(byte)
            return bytes(encoded)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0

    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("protobuf varint is too large")


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_uint_field(field_number: int, value: int) -> bytes:
    return _encode_key(field_number, WIRE_VARINT) + _encode_varint(value)


def _encode_bool_field(field_number: int, value: bool) -> bytes:
    return _encode_uint_field(field_number, 1 if value else 0)


def _encode_double_field(field_number: int, value: float) -> bytes:
    return _encode_key(field_number, WIRE_64BIT) + DOUBLE_LE.pack(value)


def _encode_string_field(field_number: int, value: str) -> bytes:
    raw = value.encode("utf-8")
    return _encode_key(field_number, WIRE_LENGTH_DELIMITED) + _encode_varint(len(raw)) + raw


def _encode_bytes_field(field_number: int, value: bytes) -> bytes:
    return _encode_key(field_number, WIRE_LENGTH_DELIMITED) + _encode_varint(len(value)) + value


def _encode_message_field(field_number: int, payload: bytes) -> bytes:
    return _encode_key(field_number, WIRE_LENGTH_DELIMITED) + _encode_varint(len(payload)) + payload


def _parse_fields(payload: bytes) -> list[tuple[int, int, int | float | bytes]]:
    fields: list[tuple[int, int, int | float | bytes]] = []
    offset = 0

    while offset < len(payload):
        key, offset = _decode_varint(payload, offset)
        field_number = key >> 3
        wire_type = key & 0x07

        if wire_type == WIRE_VARINT:
            value, offset = _decode_varint(payload, offset)
        elif wire_type == WIRE_64BIT:
            value = payload[offset : offset + DOUBLE_LE.size]
            if len(value) != DOUBLE_LE.size:
                raise ValueError("truncated protobuf 64-bit field payload")
            value = DOUBLE_LE.unpack(value)[0]
            offset += DOUBLE_LE.size
        elif wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _decode_varint(payload, offset)
            value = payload[offset : offset + length]
            if len(value) != length:
                raise ValueError("truncated protobuf field payload")
            offset += length
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")

        fields.append((field_number, wire_type, value))

    return fields


def _require_varint(wire_type: int, value: int | float | bytes) -> int:
    if wire_type != WIRE_VARINT or not isinstance(value, int):
        raise ValueError("expected protobuf varint field")
    return value


def _require_bool(wire_type: int, value: int | float | bytes) -> bool:
    return bool(_require_varint(wire_type, value))


def _require_double(wire_type: int, value: int | float | bytes) -> float:
    if wire_type != WIRE_64BIT or not isinstance(value, float):
        raise ValueError("expected protobuf 64-bit floating-point field")
    return value


def _require_bytes(wire_type: int, value: int | float | bytes) -> bytes:
    if wire_type != WIRE_LENGTH_DELIMITED or not isinstance(value, bytes):
        raise ValueError("expected protobuf length-delimited field")
    return value


def _parse_hardware_profile(payload: bytes) -> HardwareProfile:
    values: dict[int, int | str] = {}
    for field_number, wire_type, value in _parse_fields(payload):
        if wire_type == WIRE_VARINT:
            values[field_number] = _require_varint(wire_type, value)
        elif wire_type == WIRE_LENGTH_DELIMITED:
            values[field_number] = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return HardwareProfile(
        hostname=str(values.get(1, "")),
        local_ip=str(values.get(2, "")),
        mac_address=str(values.get(3, "")),
        system=str(values.get(4, "")),
        release=str(values.get(5, "")),
        machine=str(values.get(6, "")),
        processor=str(values.get(7, "")),
        logical_cpu_count=int(values.get(8, 0)),
        memory_bytes=int(values.get(9, 0)),
    )


def _encode_hardware_profile(hardware: HardwareProfile) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, hardware.hostname),
            _encode_string_field(2, hardware.local_ip),
            _encode_string_field(3, hardware.mac_address),
            _encode_string_field(4, hardware.system),
            _encode_string_field(5, hardware.release),
            _encode_string_field(6, hardware.machine),
            _encode_string_field(7, hardware.processor),
            _encode_uint_field(8, hardware.logical_cpu_count),
            _encode_uint_field(9, hardware.memory_bytes),
        ]
    )


def _parse_compute_hardware_performance(payload: bytes) -> ComputeHardwarePerformance:
    hardware_type = ""
    effective_gflops = 0.0
    rank = 0

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            hardware_type = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            effective_gflops = _require_double(wire_type, value)
        elif field_number == 3:
            rank = _require_varint(wire_type, value)

    return ComputeHardwarePerformance(
        hardware_type=hardware_type,
        effective_gflops=effective_gflops,
        rank=rank,
    )


def _encode_compute_hardware_performance(payload: ComputeHardwarePerformance) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.hardware_type),
            _encode_double_field(2, payload.effective_gflops),
            _encode_uint_field(3, payload.rank),
        ]
    )


def _parse_method_performance_summary(payload: bytes) -> MethodPerformanceSummary:
    method = ""
    hardware_count = 0
    ranked_hardware: list[ComputeHardwarePerformance] = []

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            method = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            hardware_count = _require_varint(wire_type, value)
        elif field_number == 3:
            ranked_hardware.append(_parse_compute_hardware_performance(_require_bytes(wire_type, value)))

    return MethodPerformanceSummary(
        method=method,
        hardware_count=hardware_count,
        ranked_hardware=ranked_hardware,
    )


def _encode_method_performance_summary(payload: MethodPerformanceSummary) -> bytes:
    parts = [
        _encode_string_field(1, payload.method),
        _encode_uint_field(2, payload.hardware_count),
    ]
    parts.extend(
        _encode_message_field(3, _encode_compute_hardware_performance(hardware))
        for hardware in payload.ranked_hardware
    )
    return b"".join(parts)


def _parse_compute_performance_summary(payload: bytes) -> ComputePerformanceSummary:
    hardware_count = 0
    ranked_hardware: list[ComputeHardwarePerformance] = []
    method_summaries: list[MethodPerformanceSummary] = []

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            hardware_count = _require_varint(wire_type, value)
        elif field_number == 2:
            ranked_hardware.append(_parse_compute_hardware_performance(_require_bytes(wire_type, value)))
        elif field_number == 3:
            method_summaries.append(_parse_method_performance_summary(_require_bytes(wire_type, value)))

    return ComputePerformanceSummary(
        hardware_count=hardware_count,
        ranked_hardware=ranked_hardware,
        method_summaries=method_summaries,
    )


def _encode_compute_performance_summary(payload: ComputePerformanceSummary) -> bytes:
    parts = [_encode_uint_field(1, payload.hardware_count)]
    parts.extend(
        _encode_message_field(2, _encode_compute_hardware_performance(hardware))
        for hardware in payload.ranked_hardware
    )
    parts.extend(
        _encode_message_field(3, _encode_method_performance_summary(method_summary))
        for method_summary in payload.method_summaries
    )
    return b"".join(parts)


def _parse_register_worker(payload: bytes) -> RegisterWorker:
    node_name = ""
    hardware = HardwareProfile("", "", "", "", "", "", "", 0, 0)
    performance = ComputePerformanceSummary()

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            node_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            hardware = _parse_hardware_profile(_require_bytes(wire_type, value))
        elif field_number == 3:
            performance = _parse_compute_performance_summary(_require_bytes(wire_type, value))

    return RegisterWorker(node_name=node_name, hardware=hardware, performance=performance)


def _encode_register_worker(payload: RegisterWorker) -> bytes:
    parts = [
        _encode_string_field(1, payload.node_name),
        _encode_message_field(2, _encode_hardware_profile(payload.hardware)),
    ]
    if payload.performance.hardware_count > 0 or payload.performance.ranked_hardware:
        parts.append(_encode_message_field(3, _encode_compute_performance_summary(payload.performance)))
    return b"".join(parts)


def _parse_register_ok(payload: bytes) -> RegisterOk:
    main_node_name = ""
    main_node_ip = ""
    main_node_port = 0
    node_id = ""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            main_node_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            main_node_ip = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            main_node_port = _require_varint(wire_type, value)
        elif field_number == 4:
            node_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return RegisterOk(
        main_node_name=main_node_name,
        main_node_ip=main_node_ip,
        main_node_port=main_node_port,
        node_id=node_id,
    )


def _encode_register_ok(payload: RegisterOk) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.main_node_name),
            _encode_string_field(2, payload.main_node_ip),
            _encode_uint_field(3, payload.main_node_port),
            _encode_string_field(4, payload.node_id),
        ]
    )


def _parse_heartbeat(payload: bytes) -> Heartbeat:
    main_node_name = ""
    unix_time_ms = 0

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            main_node_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            unix_time_ms = _require_varint(wire_type, value)

    return Heartbeat(main_node_name=main_node_name, unix_time_ms=unix_time_ms)


def _encode_heartbeat(payload: Heartbeat) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.main_node_name),
            _encode_uint_field(2, payload.unix_time_ms),
        ]
    )


def _parse_heartbeat_ok(payload: bytes) -> HeartbeatOk:
    node_name = ""
    heartbeat_unix_time_ms = 0
    received_unix_time_ms = 0

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            node_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            heartbeat_unix_time_ms = _require_varint(wire_type, value)
        elif field_number == 3:
            received_unix_time_ms = _require_varint(wire_type, value)

    return HeartbeatOk(
        node_name=node_name,
        heartbeat_unix_time_ms=heartbeat_unix_time_ms,
        received_unix_time_ms=received_unix_time_ms,
    )


def _encode_heartbeat_ok(payload: HeartbeatOk) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.node_name),
            _encode_uint_field(2, payload.heartbeat_unix_time_ms),
            _encode_uint_field(3, payload.received_unix_time_ms),
        ]
    )


def _parse_client_join(payload: bytes) -> ClientJoin:
    client_name = ""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            client_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return ClientJoin(client_name=client_name)


def _encode_client_join(payload: ClientJoin) -> bytes:
    return _encode_string_field(1, payload.client_name)


def _parse_client_request(payload: bytes) -> ClientRequest:
    request_id = ""
    client_name = ""
    method = ""
    object_id = ""
    stream_id = ""
    timestamp_ms = 0
    vector_length = 0
    vector_data = b""
    iteration_count = 1

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            client_name = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            method = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            object_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 5:
            stream_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 6:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 7:
            vector_length = _require_varint(wire_type, value)
        elif field_number == 8:
            vector_data = _require_bytes(wire_type, value)
        elif field_number == 9:
            iteration_count = _require_varint(wire_type, value)

    return ClientRequest(
        request_id=request_id,
        client_name=client_name,
        method=method,
        object_id=object_id,
        stream_id=stream_id,
        timestamp_ms=timestamp_ms,
        vector_length=vector_length,
        vector_data=vector_data,
        iteration_count=iteration_count,
    )


def _encode_client_request(payload: ClientRequest) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.client_name),
            _encode_string_field(3, payload.method),
            _encode_string_field(4, payload.object_id),
            _encode_string_field(5, payload.stream_id),
            _encode_uint_field(6, payload.timestamp_ms),
            _encode_uint_field(7, payload.vector_length),
            _encode_bytes_field(8, payload.vector_data),
            _encode_uint_field(9, payload.iteration_count),
        ]
    )


def _parse_client_response(payload: bytes) -> ClientResponse:
    request_id = ""
    method = ""
    object_id = ""
    stream_id = ""
    timestamp_ms = 0
    status_code = 0
    error_message = ""
    output_length = 0
    output_vector = b""
    worker_count = 0
    client_count = 0
    client_id = ""
    iteration_count = 1
    result_artifact_id = ""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            method = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            object_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            stream_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 5:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 6:
            status_code = _require_varint(wire_type, value)
        elif field_number == 7:
            error_message = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 8:
            output_length = _require_varint(wire_type, value)
        elif field_number == 9:
            output_vector = _require_bytes(wire_type, value)
        elif field_number == 10:
            worker_count = _require_varint(wire_type, value)
        elif field_number == 11:
            client_count = _require_varint(wire_type, value)
        elif field_number == 12:
            client_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 13:
            iteration_count = _require_varint(wire_type, value)
        elif field_number == 14:
            result_artifact_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return ClientResponse(
        request_id=request_id,
        method=method,
        object_id=object_id,
        stream_id=stream_id,
        timestamp_ms=timestamp_ms,
        status_code=status_code,
        error_message=error_message,
        output_length=output_length,
        output_vector=output_vector,
        worker_count=worker_count,
        client_count=client_count,
        client_id=client_id,
        iteration_count=iteration_count,
        result_artifact_id=result_artifact_id,
    )


def _encode_client_response(payload: ClientResponse) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.method),
            _encode_string_field(3, payload.object_id),
            _encode_string_field(4, payload.stream_id),
            _encode_uint_field(5, payload.timestamp_ms),
            _encode_uint_field(6, payload.status_code),
            _encode_string_field(7, payload.error_message),
            _encode_uint_field(8, payload.output_length),
            _encode_bytes_field(9, payload.output_vector),
            _encode_uint_field(10, payload.worker_count),
            _encode_uint_field(11, payload.client_count),
            _encode_string_field(12, payload.client_id),
            _encode_uint_field(13, payload.iteration_count),
            _encode_string_field(14, payload.result_artifact_id),
        ]
    )


def _parse_task_assign(payload: bytes) -> TaskAssign:
    request_id = ""
    node_id = ""
    task_id = ""
    method = ""
    object_id = ""
    stream_id = ""
    timestamp_ms = 0
    row_start = 0
    row_end = 0
    vector_length = 0
    vector_data = b""
    iteration_count = 1
    start_oc = 0
    end_oc = 0
    tensor_h = 0
    tensor_w = 0
    channels_in = 0
    channels_out = 0
    kernel_size = 0
    padding = 0
    stride = 1
    weight_data = b""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            node_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            task_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            method = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 5:
            object_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 6:
            stream_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 7:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 8:
            row_start = _require_varint(wire_type, value)
        elif field_number == 9:
            row_end = _require_varint(wire_type, value)
        elif field_number == 10:
            vector_length = _require_varint(wire_type, value)
        elif field_number == 11:
            vector_data = _require_bytes(wire_type, value)
        elif field_number == 12:
            iteration_count = _require_varint(wire_type, value)
        elif field_number == 13:
            start_oc = _require_varint(wire_type, value)
        elif field_number == 14:
            end_oc = _require_varint(wire_type, value)
        elif field_number == 15:
            tensor_h = _require_varint(wire_type, value)
        elif field_number == 16:
            tensor_w = _require_varint(wire_type, value)
        elif field_number == 17:
            channels_in = _require_varint(wire_type, value)
        elif field_number == 18:
            channels_out = _require_varint(wire_type, value)
        elif field_number == 19:
            kernel_size = _require_varint(wire_type, value)
        elif field_number == 20:
            padding = _require_varint(wire_type, value)
        elif field_number == 21:
            stride = _require_varint(wire_type, value)
        elif field_number == 22:
            weight_data = _require_bytes(wire_type, value)

    return TaskAssign(
        request_id=request_id,
        node_id=node_id,
        task_id=task_id,
        method=method,
        object_id=object_id,
        stream_id=stream_id,
        timestamp_ms=timestamp_ms,
        row_start=row_start,
        row_end=row_end,
        vector_length=vector_length,
        vector_data=vector_data,
        iteration_count=iteration_count,
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
    )


def _encode_task_assign(payload: TaskAssign) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.node_id),
            _encode_string_field(3, payload.task_id),
            _encode_string_field(4, payload.method),
            _encode_string_field(5, payload.object_id),
            _encode_string_field(6, payload.stream_id),
            _encode_uint_field(7, payload.timestamp_ms),
            _encode_uint_field(8, payload.row_start),
            _encode_uint_field(9, payload.row_end),
            _encode_uint_field(10, payload.vector_length),
            _encode_bytes_field(11, payload.vector_data),
            _encode_uint_field(12, payload.iteration_count),
            _encode_uint_field(13, payload.start_oc),
            _encode_uint_field(14, payload.end_oc),
            _encode_uint_field(15, payload.tensor_h),
            _encode_uint_field(16, payload.tensor_w),
            _encode_uint_field(17, payload.channels_in),
            _encode_uint_field(18, payload.channels_out),
            _encode_uint_field(19, payload.kernel_size),
            _encode_uint_field(20, payload.padding),
            _encode_uint_field(21, payload.stride),
            _encode_bytes_field(22, payload.weight_data),
        ]
    )


def _parse_task_accept(payload: bytes) -> TaskAccept:
    request_id = ""
    node_id = ""
    task_id = ""
    timestamp_ms = 0
    status_code = 0

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            node_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            task_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 5:
            status_code = _require_varint(wire_type, value)

    return TaskAccept(
        request_id=request_id,
        node_id=node_id,
        task_id=task_id,
        timestamp_ms=timestamp_ms,
        status_code=status_code,
    )


def _encode_task_accept(payload: TaskAccept) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.node_id),
            _encode_string_field(3, payload.task_id),
            _encode_uint_field(4, payload.timestamp_ms),
            _encode_uint_field(5, payload.status_code),
        ]
    )


def _parse_task_fail(payload: bytes) -> TaskFail:
    request_id = ""
    node_id = ""
    task_id = ""
    timestamp_ms = 0
    status_code = 0
    error_message = ""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            node_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            task_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 5:
            status_code = _require_varint(wire_type, value)
        elif field_number == 6:
            error_message = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return TaskFail(
        request_id=request_id,
        node_id=node_id,
        task_id=task_id,
        timestamp_ms=timestamp_ms,
        status_code=status_code,
        error_message=error_message,
    )


def _encode_task_fail(payload: TaskFail) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.node_id),
            _encode_string_field(3, payload.task_id),
            _encode_uint_field(4, payload.timestamp_ms),
            _encode_uint_field(5, payload.status_code),
            _encode_string_field(6, payload.error_message),
        ]
    )


def _parse_task_result(payload: bytes) -> TaskResult:
    request_id = ""
    node_id = ""
    task_id = ""
    timestamp_ms = 0
    status_code = 0
    row_start = 0
    row_end = 0
    output_length = 0
    output_vector = b""
    iteration_count = 1
    start_oc = 0
    end_oc = 0
    output_h = 0
    output_w = 0
    result_artifact_id = ""

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            request_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 2:
            node_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 3:
            task_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")
        elif field_number == 4:
            timestamp_ms = _require_varint(wire_type, value)
        elif field_number == 5:
            status_code = _require_varint(wire_type, value)
        elif field_number == 6:
            row_start = _require_varint(wire_type, value)
        elif field_number == 7:
            row_end = _require_varint(wire_type, value)
        elif field_number == 8:
            output_length = _require_varint(wire_type, value)
        elif field_number == 9:
            output_vector = _require_bytes(wire_type, value)
        elif field_number == 10:
            iteration_count = _require_varint(wire_type, value)
        elif field_number == 11:
            start_oc = _require_varint(wire_type, value)
        elif field_number == 12:
            end_oc = _require_varint(wire_type, value)
        elif field_number == 13:
            output_h = _require_varint(wire_type, value)
        elif field_number == 14:
            output_w = _require_varint(wire_type, value)
        elif field_number == 15:
            result_artifact_id = _require_bytes(wire_type, value).decode("utf-8", errors="replace")

    return TaskResult(
        request_id=request_id,
        node_id=node_id,
        task_id=task_id,
        timestamp_ms=timestamp_ms,
        status_code=status_code,
        row_start=row_start,
        row_end=row_end,
        output_length=output_length,
        output_vector=output_vector,
        iteration_count=iteration_count,
        start_oc=start_oc,
        end_oc=end_oc,
        output_h=output_h,
        output_w=output_w,
        result_artifact_id=result_artifact_id,
    )


def _encode_task_result(payload: TaskResult) -> bytes:
    return b"".join(
        [
            _encode_string_field(1, payload.request_id),
            _encode_string_field(2, payload.node_id),
            _encode_string_field(3, payload.task_id),
            _encode_uint_field(4, payload.timestamp_ms),
            _encode_uint_field(5, payload.status_code),
            _encode_uint_field(6, payload.row_start),
            _encode_uint_field(7, payload.row_end),
            _encode_uint_field(8, payload.output_length),
            _encode_bytes_field(9, payload.output_vector),
            _encode_uint_field(10, payload.iteration_count),
            _encode_uint_field(11, payload.start_oc),
            _encode_uint_field(12, payload.end_oc),
            _encode_uint_field(13, payload.output_h),
            _encode_uint_field(14, payload.output_w),
            _encode_string_field(15, payload.result_artifact_id),
        ]
    )


@trace_function
def encode_envelope(message: RuntimeEnvelope) -> bytes:
    """Serialize a runtime message into protobuf bytes."""

    parts = [_encode_uint_field(1, int(message.kind))]

    if message.kind == MessageKind.REGISTER_WORKER:
        if message.register_worker is None:
            raise ValueError("REGISTER_WORKER envelope missing payload")
        parts.append(_encode_message_field(2, _encode_register_worker(message.register_worker)))
    elif message.kind == MessageKind.REGISTER_OK:
        if message.register_ok is None:
            raise ValueError("REGISTER_OK envelope missing payload")
        parts.append(_encode_message_field(3, _encode_register_ok(message.register_ok)))
    elif message.kind == MessageKind.HEARTBEAT:
        if message.heartbeat is None:
            raise ValueError("HEARTBEAT envelope missing payload")
        parts.append(_encode_message_field(4, _encode_heartbeat(message.heartbeat)))
    elif message.kind == MessageKind.HEARTBEAT_OK:
        if message.heartbeat_ok is None:
            raise ValueError("HEARTBEAT_OK envelope missing payload")
        parts.append(_encode_message_field(5, _encode_heartbeat_ok(message.heartbeat_ok)))
    elif message.kind == MessageKind.CLIENT_JOIN:
        if message.client_join is None:
            raise ValueError("CLIENT_JOIN envelope missing payload")
        parts.append(_encode_message_field(6, _encode_client_join(message.client_join)))
    elif message.kind == MessageKind.CLIENT_REQUEST:
        if message.client_request is None:
            raise ValueError("CLIENT_REQUEST envelope missing payload")
        parts.append(_encode_message_field(7, _encode_client_request(message.client_request)))
    elif message.kind == MessageKind.CLIENT_RESPONSE:
        if message.client_response is None:
            raise ValueError("CLIENT_RESPONSE envelope missing payload")
        parts.append(_encode_message_field(8, _encode_client_response(message.client_response)))
    elif message.kind == MessageKind.TASK_ASSIGN:
        if message.task_assign is None:
            raise ValueError("TASK_ASSIGN envelope missing payload")
        parts.append(_encode_message_field(9, _encode_task_assign(message.task_assign)))
    elif message.kind == MessageKind.TASK_ACCEPT:
        if message.task_accept is None:
            raise ValueError("TASK_ACCEPT envelope missing payload")
        parts.append(_encode_message_field(10, _encode_task_accept(message.task_accept)))
    elif message.kind == MessageKind.TASK_FAIL:
        if message.task_fail is None:
            raise ValueError("TASK_FAIL envelope missing payload")
        parts.append(_encode_message_field(11, _encode_task_fail(message.task_fail)))
    elif message.kind == MessageKind.TASK_RESULT:
        if message.task_result is None:
            raise ValueError("TASK_RESULT envelope missing payload")
        parts.append(_encode_message_field(12, _encode_task_result(message.task_result)))

    return b"".join(parts)


@trace_function
def parse_envelope(payload: bytes) -> RuntimeEnvelope:
    """Parse protobuf bytes into a runtime message."""

    kind = MessageKind.UNSPECIFIED
    register_worker = None
    register_ok = None
    heartbeat = None
    heartbeat_ok = None
    client_join = None
    client_request = None
    client_response = None
    task_assign = None
    task_accept = None
    task_fail = None
    task_result = None

    for field_number, wire_type, value in _parse_fields(payload):
        if field_number == 1:
            kind = MessageKind(_require_varint(wire_type, value))
        elif field_number == 2:
            register_worker = _parse_register_worker(_require_bytes(wire_type, value))
        elif field_number == 3:
            register_ok = _parse_register_ok(_require_bytes(wire_type, value))
        elif field_number == 4:
            heartbeat = _parse_heartbeat(_require_bytes(wire_type, value))
        elif field_number == 5:
            heartbeat_ok = _parse_heartbeat_ok(_require_bytes(wire_type, value))
        elif field_number == 6:
            client_join = _parse_client_join(_require_bytes(wire_type, value))
        elif field_number == 7:
            client_request = _parse_client_request(_require_bytes(wire_type, value))
        elif field_number == 8:
            client_response = _parse_client_response(_require_bytes(wire_type, value))
        elif field_number == 9:
            task_assign = _parse_task_assign(_require_bytes(wire_type, value))
        elif field_number == 10:
            task_accept = _parse_task_accept(_require_bytes(wire_type, value))
        elif field_number == 11:
            task_fail = _parse_task_fail(_require_bytes(wire_type, value))
        elif field_number == 12:
            task_result = _parse_task_result(_require_bytes(wire_type, value))

    return RuntimeEnvelope(
        kind=kind,
        register_worker=register_worker,
        register_ok=register_ok,
        heartbeat=heartbeat,
        heartbeat_ok=heartbeat_ok,
        client_join=client_join,
        client_request=client_request,
        client_response=client_response,
        task_assign=task_assign,
        task_accept=task_accept,
        task_fail=task_fail,
        task_result=task_result,
    )


def _recv_exactly(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("truncated framed protobuf message")
        chunks.extend(chunk)
    return bytes(chunks)


@trace_function
def send_message(sock: socket.socket, message: RuntimeEnvelope) -> None:
    """Send one length-prefixed protobuf message."""

    payload = encode_envelope(message)
    sock.sendall(FRAME_HEADER.pack(len(payload)))
    sock.sendall(payload)


@trace_function
def recv_message(sock: socket.socket, *, max_size: int) -> RuntimeEnvelope | None:
    """Receive one length-prefixed protobuf message."""

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
    """Create a compute-node registration message."""

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
    """Create a main-node registration-ack message."""

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
    """Create a main-node heartbeat message."""

    if unix_time_ms is None:
        unix_time_ms = int(time.time() * 1000)

    return RuntimeEnvelope(
        kind=MessageKind.HEARTBEAT,
        heartbeat=Heartbeat(main_node_name=main_node_name, unix_time_ms=unix_time_ms),
    )


@trace_function
def build_heartbeat_ok(node_name: str, heartbeat_unix_time_ms: int, received_unix_time_ms: int | None = None) -> RuntimeEnvelope:
    """Create a worker heartbeat acknowledgement."""

    if received_unix_time_ms is None:
        received_unix_time_ms = int(time.time() * 1000)

    return RuntimeEnvelope(
        kind=MessageKind.HEARTBEAT_OK,
        heartbeat_ok=HeartbeatOk(
            node_name=node_name,
            heartbeat_unix_time_ms=heartbeat_unix_time_ms,
            received_unix_time_ms=received_unix_time_ms,
        ),
    )


@trace_function
def build_client_join(client_name: str = SUPERWEB_CLIENT_NAME) -> RuntimeEnvelope:
    """Create a client join message."""

    return RuntimeEnvelope(kind=MessageKind.CLIENT_JOIN, client_join=ClientJoin(client_name=client_name))


@trace_function
def build_client_request(
    client_name: str,
    request_id: str,
    method: str,
    vector_data: bytes,
    *,
    object_id: str = "",
    stream_id: str = "",
    timestamp_ms: int | None = None,
    vector_length: int | None = None,
    iteration_count: int = 1,
) -> RuntimeEnvelope:
    """Create a client request message."""

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
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            vector_length=vector_length,
            vector_data=vector_data,
            iteration_count=iteration_count,
        ),
    )


@trace_function
def build_client_response(
    request_id: str,
    status_code: int,
    *,
    method: str = "",
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
    result_artifact_id: str = "",
) -> RuntimeEnvelope:
    """Create a main-node response to a client join or request."""

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
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            status_code=status_code,
            error_message=error_message,
            output_length=output_length,
            output_vector=output_vector,
            worker_count=worker_count,
            client_count=client_count,
            client_id=client_id,
            iteration_count=iteration_count,
            result_artifact_id=result_artifact_id,
        ),
    )


@trace_function
def build_task_assign(
    request_id: str,
    node_id: str,
    task_id: str,
    method: str,
    row_start: int,
    row_end: int,
    vector_data: bytes,
    *,
    object_id: str = "",
    stream_id: str = "",
    timestamp_ms: int | None = None,
    vector_length: int | None = None,
    iteration_count: int = 1,
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
) -> RuntimeEnvelope:
    """Create one task assignment message for a worker."""

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
            object_id=object_id,
            stream_id=stream_id,
            timestamp_ms=timestamp_ms,
            row_start=row_start,
            row_end=row_end,
            vector_length=vector_length,
            vector_data=vector_data,
            iteration_count=iteration_count,
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
    """Create a task-accept acknowledgement from a worker."""

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
    """Create a task-failure message from a worker."""

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
    row_start: int,
    row_end: int,
    output_vector: bytes,
    *,
    timestamp_ms: int | None = None,
    output_length: int | None = None,
    iteration_count: int = 1,
    start_oc: int = 0,
    end_oc: int = 0,
    output_h: int = 0,
    output_w: int = 0,
    result_artifact_id: str = "",
) -> RuntimeEnvelope:
    """Create a task-result message from a worker."""

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
            row_start=row_start,
            row_end=row_end,
            output_length=output_length,
            output_vector=output_vector,
            iteration_count=iteration_count,
            start_oc=start_oc,
            end_oc=end_oc,
            output_h=output_h,
            output_w=output_w,
            result_artifact_id=result_artifact_id,
        ),
    )


def describe_message_kind(kind: MessageKind) -> str:
    """Return the user-facing label for a runtime message kind."""

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
    if kind == MessageKind.CLIENT_REQUEST:
        return RUNTIME_MSG_CLIENT_REQUEST
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
    return kind.name

