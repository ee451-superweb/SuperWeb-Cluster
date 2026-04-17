"""Encode and decode framed data-plane messages for large artifact transfer.

Use this module when the control plane has already exchanged an
``ArtifactDescriptor`` and two peers now need a simple binary framing format to
request, announce, stream, and validate artifact bytes over TCP.
"""

from __future__ import annotations

import struct

from wire.internal_protocol.data_plane import (
    ArtifactChunkFrame,
    ArtifactEndFrame,
    ArtifactErrorFrame,
    ArtifactInitFrame,
    ArtifactRequestFrame,
    DataPlaneMessageType,
)

MAGIC = b"SWAD"
PROTOCOL_VERSION = 1

# Keep this framing identical to the standalone client implementation.
REQUEST_HEADER = struct.Struct("!4sBBI")
INIT_HEADER = struct.Struct("!4sBBQIII")
CHUNK_HEADER = struct.Struct("!4sBBQI")
END_HEADER = struct.Struct("!4sBBQ")
ERROR_HEADER = struct.Struct("!4sBBI")

# Backward-compatible numeric aliases for existing transport code.
MSG_REQUEST = int(DataPlaneMessageType.REQUEST)
MSG_INIT = int(DataPlaneMessageType.INIT)
MSG_CHUNK = int(DataPlaneMessageType.CHUNK)
MSG_END = int(DataPlaneMessageType.END)
MSG_ERROR = int(DataPlaneMessageType.ERROR)


def _validate_header(magic: bytes, version: int, expected_type: DataPlaneMessageType) -> None:
    """Validate one data-plane frame header before decoding its payload."""
    if magic != MAGIC:
        raise ValueError("invalid data-plane magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported data-plane version: {version}")
    if expected_type not in set(DataPlaneMessageType):
        raise ValueError(f"invalid expected message type: {expected_type}")


def encode_request(artifact_id: str) -> bytes:
    """Serialize one artifact request frame."""
    frame = ArtifactRequestFrame(artifact_id=artifact_id)
    raw = frame.artifact_id.encode("utf-8")
    return REQUEST_HEADER.pack(MAGIC, PROTOCOL_VERSION, int(DataPlaneMessageType.REQUEST), len(raw)) + raw


def decode_request(header: bytes, payload: bytes) -> str:
    """Decode one artifact request frame."""
    magic, version, message_type, name_len = REQUEST_HEADER.unpack(header)
    _validate_header(magic, version, DataPlaneMessageType.REQUEST)
    if message_type != int(DataPlaneMessageType.REQUEST):
        raise ValueError("expected REQUEST message")
    if len(payload) != name_len:
        raise ValueError("artifact request payload length mismatch")
    return ArtifactRequestFrame(artifact_id=payload.decode("utf-8")).artifact_id


def encode_init(*, size_bytes: int, chunk_size: int, checksum: str, content_type: str) -> bytes:
    """Serialize one init frame describing the artifact that will be streamed."""
    frame = ArtifactInitFrame(
        size_bytes=size_bytes,
        chunk_size=chunk_size,
        checksum=checksum,
        content_type=content_type,
    )
    checksum_raw = frame.checksum.encode("utf-8")
    content_type_raw = frame.content_type.encode("utf-8")
    return (
        INIT_HEADER.pack(
            MAGIC,
            PROTOCOL_VERSION,
            int(DataPlaneMessageType.INIT),
            frame.size_bytes,
            frame.chunk_size,
            len(checksum_raw),
            len(content_type_raw),
        )
        + checksum_raw
        + content_type_raw
    )


def decode_init(header: bytes, payload: bytes) -> tuple[int, int, str, str]:
    """Decode one init frame describing the artifact that will be streamed."""
    magic, version, message_type, size_bytes, chunk_size, checksum_len, content_type_len = INIT_HEADER.unpack(header)
    _validate_header(magic, version, DataPlaneMessageType.INIT)
    if message_type != int(DataPlaneMessageType.INIT):
        raise ValueError("expected INIT message")
    if len(payload) != checksum_len + content_type_len:
        raise ValueError("artifact init payload length mismatch")
    checksum_raw = payload[:checksum_len]
    content_type_raw = payload[checksum_len:]
    frame = ArtifactInitFrame(
        size_bytes=size_bytes,
        chunk_size=chunk_size,
        checksum=checksum_raw.decode("utf-8"),
        content_type=content_type_raw.decode("utf-8"),
    )
    return frame.size_bytes, frame.chunk_size, frame.checksum, frame.content_type


def encode_chunk(*, offset: int, data: bytes) -> bytes:
    """Serialize one artifact chunk frame."""
    frame = ArtifactChunkFrame(offset=offset, data=data)
    return CHUNK_HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(DataPlaneMessageType.CHUNK),
        frame.offset,
        len(frame.data),
    ) + frame.data


def decode_chunk(header: bytes, payload: bytes) -> tuple[int, bytes]:
    """Decode one artifact chunk frame."""
    magic, version, message_type, offset, length = CHUNK_HEADER.unpack(header)
    _validate_header(magic, version, DataPlaneMessageType.CHUNK)
    if message_type != int(DataPlaneMessageType.CHUNK):
        raise ValueError("expected CHUNK message")
    if len(payload) != length:
        raise ValueError("artifact chunk payload length mismatch")
    frame = ArtifactChunkFrame(offset=offset, data=payload)
    return frame.offset, frame.data


def encode_end(*, size_bytes: int) -> bytes:
    """Serialize one end frame signaling transfer completion."""
    frame = ArtifactEndFrame(size_bytes=size_bytes)
    return END_HEADER.pack(MAGIC, PROTOCOL_VERSION, int(DataPlaneMessageType.END), frame.size_bytes)


def decode_end(header: bytes) -> int:
    """Decode one end frame signaling transfer completion."""
    magic, version, message_type, size_bytes = END_HEADER.unpack(header)
    _validate_header(magic, version, DataPlaneMessageType.END)
    if message_type != int(DataPlaneMessageType.END):
        raise ValueError("expected END message")
    return ArtifactEndFrame(size_bytes=size_bytes).size_bytes


def encode_error(message: str) -> bytes:
    """Serialize one error frame terminating the transfer."""
    frame = ArtifactErrorFrame(message=message)
    raw = frame.message.encode("utf-8")
    return ERROR_HEADER.pack(MAGIC, PROTOCOL_VERSION, int(DataPlaneMessageType.ERROR), len(raw)) + raw


def decode_error(header: bytes, payload: bytes) -> str:
    """Decode one error frame terminating the transfer."""
    magic, version, message_type, length = ERROR_HEADER.unpack(header)
    _validate_header(magic, version, DataPlaneMessageType.ERROR)
    if message_type != int(DataPlaneMessageType.ERROR):
        raise ValueError("expected ERROR message")
    if len(payload) != length:
        raise ValueError("artifact error payload length mismatch")
    return ArtifactErrorFrame(message=payload.decode("utf-8")).message
