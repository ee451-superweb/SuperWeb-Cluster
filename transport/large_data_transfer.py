"""Serve and fetch large artifacts over a simple chunked TCP data plane.

Use this module when the control plane has already exchanged an
``ArtifactDescriptor`` and two peers now need to publish or download the
artifact bytes themselves with checksum verification.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from wire.internal_protocol.data_plane_codec import (
    CHUNK_HEADER,
    END_HEADER,
    ERROR_HEADER,
    INIT_HEADER,
    REQUEST_HEADER,
    decode_chunk,
    decode_end,
    decode_error,
    decode_init,
    decode_request,
    encode_chunk,
    encode_end,
    encode_error,
    encode_init,
    encode_request,
)
from wire.external_protocol.data_plane import ArtifactDescriptor


def _recv_exactly(sock: socket.socket, size: int) -> bytes:
    """Use this internal helper when one framed data-plane read needs exact bytes.

    Args: sock connected TCP socket and size exact byte count to read.
    Returns: The requested bytes, or raises when the peer closes mid-frame.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(
                "socket.recv() must return bytes-like data, "
                f"got {type(chunk).__name__}"
            )
        chunk = bytes(chunk)
        if not chunk:
            raise ConnectionError("peer closed data-plane socket mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _sha256_path(path: Path, *, chunk_size: int) -> str:
    """Use this when publishing an artifact descriptor that needs a checksum.

    Args: path local file path to hash and chunk_size read size used while hashing.
    Returns: The hex SHA-256 digest for the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class PublishedArtifact:
    """Bundle one published descriptor together with its backing local file path."""

    descriptor: ArtifactDescriptor
    local_path: Path


class LargeDataTransferServer:
    """Serve published artifacts over a simple TCP chunked protocol."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        resolve_artifact: Callable[[str], PublishedArtifact | None],
        chunk_size: int,
    ) -> None:
        """Use this when one process needs to expose artifacts over TCP.

        Args: host/port listener endpoint, resolve_artifact callback to map ids to files, chunk_size preferred read size while streaming.
        Returns: None after the server stores its listen settings and callbacks.
        """
        self.host = host
        self.port = port
        self.resolve_artifact = resolve_artifact
        self.chunk_size = chunk_size
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Use this before any artifact descriptor is handed to a remote peer.

        Args: self server whose listening socket and accept thread should be started.
        Returns: None after the TCP listener is active.
        """
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen()
        sock.settimeout(1.0)
        self.port = sock.getsockname()[1]
        self._sock = sock
        self._thread = threading.Thread(target=self._serve_loop, name="artifact-data-plane", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Use this during shutdown to stop accepting new data-plane connections.

        Args: self server whose listener should be closed.
        Returns: None after the stop event is set and the socket is closed.
        """
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _serve_loop(self) -> None:
        """Use this background loop to accept and dispatch data-plane connections.

        Args: self server whose listening socket is already active.
        Returns: None after shutdown stops the accept loop.
        """
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                raise
            threading.Thread(target=self._serve_connection, args=(conn,), daemon=True).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        """Use this per-connection worker to serve one artifact request over TCP.

        Args: conn accepted data-plane TCP socket from one fetching peer.
        Returns: None after the request is served or an error is reported.
        """
        try:
            request_header = _recv_exactly(conn, REQUEST_HEADER.size)
            _, _, _, artifact_id_len = REQUEST_HEADER.unpack(request_header)
            artifact_id = decode_request(request_header, _recv_exactly(conn, artifact_id_len))
            artifact = self.resolve_artifact(artifact_id)
            if artifact is None or not artifact.local_path.exists():
                conn.sendall(encode_error(f"artifact not found: {artifact_id}"))
                return

            conn.sendall(
                encode_init(
                    size_bytes=artifact.descriptor.size_bytes,
                    chunk_size=artifact.descriptor.chunk_size,
                    checksum=artifact.descriptor.checksum,
                    content_type=artifact.descriptor.content_type,
                )
            )
            offset = 0
            with artifact.local_path.open("rb") as handle:
                while True:
                    chunk = handle.read(self.chunk_size)
                    if not chunk:
                        break
                    conn.sendall(encode_chunk(offset=offset, data=chunk))
                    offset += len(chunk)
            conn.sendall(encode_end(size_bytes=artifact.descriptor.size_bytes))
        finally:
            try:
                conn.close()
            except OSError:
                pass


def fetch_artifact_to_file(
    descriptor: ArtifactDescriptor,
    destination_path: Path,
    *,
    timeout: float,
) -> Path:
    """Use this when a remote artifact should be downloaded and verified on disk.

    Args: descriptor remote artifact descriptor, destination_path local destination path, timeout socket timeout in seconds.
    Returns: The destination path after size and checksum validation succeed.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    digest = hashlib.sha256()
    bytes_written = 0

    with socket.create_connection((descriptor.transfer_host, descriptor.transfer_port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(encode_request(descriptor.artifact_id))

        header = _recv_exactly(sock, 6)
        message_type = header[5]
        if message_type == 2:
            rest = _recv_exactly(sock, INIT_HEADER.size - 6)
            init_header = header + rest
            _, server_chunk_size, server_checksum, _content_type = _read_init_payload(sock, init_header)
            chunk_size = server_chunk_size or descriptor.chunk_size
        elif message_type == 5:
            rest = _recv_exactly(sock, ERROR_HEADER.size - 6)
            error_header = header + rest
            _, _, _, length = ERROR_HEADER.unpack(error_header)
            raise RuntimeError(decode_error(error_header, _recv_exactly(sock, length)))
        else:
            raise ValueError("unexpected first data-plane message type")

        with tmp_path.open("wb") as handle:
            while True:
                header = _recv_exactly(sock, 6)
                message_type = header[5]
                if message_type == 3:
                    rest = _recv_exactly(sock, CHUNK_HEADER.size - 6)
                    chunk_header = header + rest
                    _, _, _, _offset, length = CHUNK_HEADER.unpack(chunk_header)
                    _offset, payload = decode_chunk(chunk_header, _recv_exactly(sock, length))
                    handle.write(payload)
                    digest.update(payload)
                    bytes_written += len(payload)
                elif message_type == 4:
                    rest = _recv_exactly(sock, END_HEADER.size - 6)
                    end_header = header + rest
                    total_size = decode_end(end_header)
                    if total_size != bytes_written:
                        raise ValueError("artifact end size mismatch")
                    break
                elif message_type == 5:
                    rest = _recv_exactly(sock, ERROR_HEADER.size - 6)
                    error_header = header + rest
                    _, _, _, length = ERROR_HEADER.unpack(error_header)
                    raise RuntimeError(decode_error(error_header, _recv_exactly(sock, length)))
                else:
                    raise ValueError("unexpected data-plane message type")

    checksum = digest.hexdigest()
    expected_checksum = server_checksum or descriptor.checksum
    if checksum != expected_checksum:
        raise ValueError("artifact checksum mismatch")
    if bytes_written != descriptor.size_bytes:
        raise ValueError("artifact size mismatch")
    os.replace(tmp_path, destination_path)
    return destination_path


def _read_init_payload(sock: socket.socket, init_header: bytes) -> tuple[int, int, str, str]:
    """Use this internal helper after the first INIT header has been read.

    Args: sock connected TCP socket and init_header already-read INIT header bytes.
    Returns: A tuple of ``(size_bytes, chunk_size, checksum, content_type)``.
    """
    _, _, _, size_bytes, chunk_size, checksum_len, content_type_len = INIT_HEADER.unpack(init_header)
    payload = _recv_exactly(sock, checksum_len + content_type_len)
    _, _, checksum, content_type = decode_init(init_header, payload)
    return size_bytes, chunk_size, checksum, content_type


def publish_file_descriptor(
    *,
    artifact_id: str,
    local_path: Path,
    public_host: str,
    public_port: int,
    producer_node_id: str,
    chunk_size: int,
    content_type: str = "application/octet-stream",
) -> PublishedArtifact:
    """Use this when one local file should become a published artifact descriptor.

    Args: artifact_id protocol id, local_path backing file, public_host/public_port remote endpoint, producer_node_id source node id, chunk_size advertised chunk size, content_type optional MIME-like label.
    Returns: A PublishedArtifact containing both the descriptor and local file path.
    """
    return PublishedArtifact(
        descriptor=ArtifactDescriptor(
            artifact_id=artifact_id,
            content_type=content_type,
            size_bytes=local_path.stat().st_size,
            checksum=_sha256_path(local_path, chunk_size=chunk_size),
            producer_node_id=producer_node_id,
            transfer_host=public_host,
            transfer_port=public_port,
            chunk_size=chunk_size,
            ready=True,
        ),
        local_path=local_path,
    )
