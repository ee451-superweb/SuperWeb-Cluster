"""Serve and fetch large artifacts over a simple chunked TCP data plane.

Use this module when the control plane has already exchanged an
``ArtifactDescriptor`` and two peers now need to publish or download the
artifact bytes themselves with checksum verification.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from wire.internal_protocol.data_plane_codec import (
    CHUNK_HEADER,
    DELIVER_HEADER,
    DOWNLOAD_REQUEST_HEADER,
    END_HEADER,
    ERROR_HEADER,
    INIT_HEADER,
    MSG_CHUNK,
    MSG_DELIVER,
    MSG_DOWNLOAD_REQUEST,
    MSG_END,
    MSG_ERROR,
    decode_chunk,
    decode_deliver,
    decode_download_request,
    decode_end,
    decode_error,
    decode_init,
    encode_chunk,
    encode_download_request,
    encode_end,
    encode_error,
    encode_init,
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


@dataclass(slots=True)
class UploadSlot:
    """One pending upload slot pre-registered by the control plane."""

    upload_id: str
    expected_size: int
    expected_checksum: str
    expected_content_type: str
    destination_path: Path
    completion_future: concurrent.futures.Future


class LargeDataTransferServer:
    """Serve published artifacts and accept client-initiated uploads over TCP."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        resolve_artifact: Callable[[str], PublishedArtifact | None],
        resolve_upload_slot: Callable[[str], UploadSlot | None] | None = None,
        consume_upload_slot: Callable[[str], None] | None = None,
        chunk_size: int,
    ) -> None:
        """Use this when one process needs to expose artifacts and accept uploads over TCP.

        Args: host/port listener endpoint, resolve_artifact callback to map ids to files, resolve_upload_slot optional callback that returns a pre-registered upload slot, consume_upload_slot optional callback invoked after a slot completes or fails, chunk_size preferred read size while streaming.
        Returns: None after the server stores its listen settings and callbacks.
        """
        self.host = host
        self.port = port
        self.resolve_artifact = resolve_artifact
        self.resolve_upload_slot = resolve_upload_slot
        self.consume_upload_slot = consume_upload_slot
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
        """Use this per-connection worker to serve one data-plane session over TCP.

        Args: conn accepted data-plane TCP socket from one fetching or uploading peer.
        Returns: None after the session closes, potentially spanning one upload and one download.
        """
        try:
            peek = _recv_exactly(conn, 6)
            message_type = peek[5]
            if message_type == int(MSG_DOWNLOAD_REQUEST):
                rest = _recv_exactly(conn, DOWNLOAD_REQUEST_HEADER.size - 6)
                self._serve_download(conn, peek + rest)
            elif message_type == int(MSG_DELIVER):
                rest = _recv_exactly(conn, DELIVER_HEADER.size - 6)
                if not self._serve_upload(conn, peek + rest):
                    return
                # After a successful upload the client may immediately send a
                # DOWNLOAD_REQUEST on the same socket to pull the eventual result
                # artifact. We wait for either that frame or a clean EOF.
                try:
                    next_peek = _recv_exactly(conn, 6)
                except ConnectionError:
                    return
                next_type = next_peek[5]
                if next_type == int(MSG_DOWNLOAD_REQUEST):
                    rest = _recv_exactly(conn, DOWNLOAD_REQUEST_HEADER.size - 6)
                    self._serve_download(conn, next_peek + rest)
                else:
                    conn.sendall(encode_error(f"unexpected post-upload message type: {next_type}"))
            else:
                conn.sendall(encode_error(f"unexpected first data-plane message type: {message_type}"))
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _serve_download(self, conn: socket.socket, request_header: bytes) -> None:
        """Use this internal helper to handle one DOWNLOAD_REQUEST-initiated artifact download."""
        _, _, _, artifact_id_len = DOWNLOAD_REQUEST_HEADER.unpack(request_header)
        artifact_id = decode_download_request(request_header, _recv_exactly(conn, artifact_id_len))
        artifact = self.resolve_artifact(artifact_id)
        if artifact is None or not artifact.local_path.exists():
            conn.sendall(encode_error(f"artifact not found: {artifact_id}"))
            return
        serve_started_at = time.perf_counter_ns()
        logger.info(
            "[DIAG] serve starting artifact_id=%s total_bytes=%s chunk_size=%s",
            artifact_id,
            artifact.descriptor.size_bytes,
            artifact.descriptor.chunk_size,
        )
        conn.sendall(
            encode_init(
                size_bytes=artifact.descriptor.size_bytes,
                chunk_size=artifact.descriptor.chunk_size,
                checksum=artifact.descriptor.checksum,
                content_type=artifact.descriptor.content_type,
            )
        )
        offset = 0
        read_ns = 0
        send_ns = 0
        progress_bucket_bytes = 64 * 1024 * 1024
        progress_next_threshold = progress_bucket_bytes
        with artifact.local_path.open("rb") as handle:
            while True:
                _t = time.perf_counter_ns()
                chunk = handle.read(self.chunk_size)
                read_ns += time.perf_counter_ns() - _t
                if not chunk:
                    break
                _t = time.perf_counter_ns()
                conn.sendall(encode_chunk(offset=offset, data=chunk))
                send_ns += time.perf_counter_ns() - _t
                offset += len(chunk)
                if offset >= progress_next_threshold:
                    elapsed_ms = (time.perf_counter_ns() - serve_started_at) // 1_000_000
                    rate_mbps = (offset / max(1, elapsed_ms)) * 1000.0 / (1024 * 1024)
                    logger.info(
                        "[DIAG] serve progress artifact_id=%s bytes=%s/%s elapsed_ms=%d rate_MBps=%.2f read_ms=%d send_ms=%d",
                        artifact_id,
                        offset,
                        artifact.descriptor.size_bytes,
                        elapsed_ms,
                        rate_mbps,
                        read_ns // 1_000_000,
                        send_ns // 1_000_000,
                    )
                    progress_next_threshold += progress_bucket_bytes
        conn.sendall(encode_end(size_bytes=artifact.descriptor.size_bytes))
        total_ms = (time.perf_counter_ns() - serve_started_at) // 1_000_000
        rate_mbps = (offset / max(1, total_ms)) * 1000.0 / (1024 * 1024)
        logger.info(
            "[DIAG] serve complete artifact_id=%s bytes=%s elapsed_ms=%d rate_MBps=%.2f read_ms=%d send_ms=%d",
            artifact_id,
            offset,
            total_ms,
            rate_mbps,
            read_ns // 1_000_000,
            send_ns // 1_000_000,
        )

    def _serve_upload(self, conn: socket.socket, deliver_header: bytes) -> bool:
        """Use this internal helper to accept a DELIVER-initiated upload stream.

        Returns True after a successful upload, False when the upload was rejected
        or aborted (the socket has already been closed in that case by an error frame).
        """
        _, _, _, _size_bytes, upload_len, checksum_len, content_type_len = DELIVER_HEADER.unpack(deliver_header)
        deliver_payload = _recv_exactly(conn, upload_len + checksum_len + content_type_len)
        frame = decode_deliver(deliver_header, deliver_payload)
        if self.resolve_upload_slot is None:
            conn.sendall(encode_error("uploads are not accepted on this endpoint"))
            return False
        slot = self.resolve_upload_slot(frame.upload_id)
        if slot is None:
            conn.sendall(encode_error(f"upload slot not registered: {frame.upload_id}"))
            return False
        if slot.expected_size != frame.size_bytes:
            self._fail_upload(conn, slot, f"upload size mismatch: expected {slot.expected_size}, got {frame.size_bytes}")
            return False
        if slot.expected_checksum and slot.expected_checksum != frame.checksum:
            self._fail_upload(conn, slot, "upload checksum mismatch")
            return False
        if slot.expected_content_type and slot.expected_content_type != frame.content_type:
            self._fail_upload(conn, slot, "upload content_type mismatch")
            return False

        slot.destination_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = slot.destination_path.with_suffix(slot.destination_path.suffix + ".tmp")
        digest = hashlib.sha256()
        bytes_written = 0
        recv_ns = 0
        write_ns = 0
        digest_ns = 0
        try:
            with tmp_path.open("wb") as handle:
                while True:
                    header = _recv_exactly(conn, 6)
                    next_type = header[5]
                    if next_type == int(MSG_CHUNK):
                        rest = _recv_exactly(conn, CHUNK_HEADER.size - 6)
                        chunk_header = header + rest
                        _, _, _, _offset, length = CHUNK_HEADER.unpack(chunk_header)
                        _t = time.perf_counter_ns()
                        raw_payload = _recv_exactly(conn, length)
                        recv_ns += time.perf_counter_ns() - _t
                        _, payload = decode_chunk(chunk_header, raw_payload)
                        _t = time.perf_counter_ns()
                        handle.write(payload)
                        write_ns += time.perf_counter_ns() - _t
                        _t = time.perf_counter_ns()
                        digest.update(payload)
                        digest_ns += time.perf_counter_ns() - _t
                        bytes_written += len(payload)
                    elif next_type == int(MSG_END):
                        rest = _recv_exactly(conn, END_HEADER.size - 6)
                        end_header = header + rest
                        total_size = decode_end(end_header)
                        if total_size != bytes_written:
                            raise ValueError("upload end size mismatch")
                        break
                    elif next_type == int(MSG_ERROR):
                        rest = _recv_exactly(conn, ERROR_HEADER.size - 6)
                        error_header = header + rest
                        _, _, _, length = ERROR_HEADER.unpack(error_header)
                        raise RuntimeError(decode_error(error_header, _recv_exactly(conn, length)))
                    else:
                        raise ValueError(f"unexpected upload message type: {next_type}")

            if bytes_written != slot.expected_size:
                raise ValueError("upload size mismatch after END")
            local_checksum = digest.hexdigest()
            if slot.expected_checksum and local_checksum != slot.expected_checksum:
                raise ValueError("upload checksum mismatch after END")
            os.replace(tmp_path, slot.destination_path)
            logger.info(
                "upload accepted upload_id=%s bytes=%s declared_checksum=%s local_checksum=%s recv_ms=%d write_ms=%d digest_ms=%d",
                frame.upload_id,
                bytes_written,
                slot.expected_checksum or "<none>",
                local_checksum,
                recv_ns // 1_000_000,
                write_ns // 1_000_000,
                digest_ns // 1_000_000,
            )
        except Exception as exc:  # noqa: BLE001 — we rethrow via the slot future
            tmp_path.unlink(missing_ok=True)
            slot.completion_future.set_exception(exc)
            if self.consume_upload_slot is not None:
                self.consume_upload_slot(frame.upload_id)
            try:
                conn.sendall(encode_error(f"upload failed: {exc}"))
            except OSError:
                pass
            return False

        slot.completion_future.set_result(slot.destination_path)
        if self.consume_upload_slot is not None:
            self.consume_upload_slot(frame.upload_id)
        return True

    def _fail_upload(self, conn: socket.socket, slot: UploadSlot, message: str) -> None:
        """Use this internal helper when a DELIVER frame fails pre-stream validation."""
        try:
            conn.sendall(encode_error(message))
        except OSError:
            pass
        slot.completion_future.set_exception(ValueError(message))
        if self.consume_upload_slot is not None:
            self.consume_upload_slot(slot.upload_id)


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
    recv_ns = 0
    write_ns = 0
    digest_ns = 0
    fetch_started_at = time.perf_counter_ns()
    progress_bucket_bytes = 64 * 1024 * 1024
    progress_next_threshold = progress_bucket_bytes
    logger.info(
        "[DIAG] fetch starting artifact_id=%s host=%s port=%s expected_bytes=%s",
        descriptor.artifact_id,
        descriptor.transfer_host,
        descriptor.transfer_port,
        descriptor.size_bytes,
    )

    with socket.create_connection((descriptor.transfer_host, descriptor.transfer_port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(encode_download_request(descriptor.artifact_id))

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
                    _t = time.perf_counter_ns()
                    raw_payload = _recv_exactly(sock, length)
                    recv_ns += time.perf_counter_ns() - _t
                    _offset, payload = decode_chunk(chunk_header, raw_payload)
                    _t = time.perf_counter_ns()
                    handle.write(payload)
                    write_ns += time.perf_counter_ns() - _t
                    _t = time.perf_counter_ns()
                    digest.update(payload)
                    digest_ns += time.perf_counter_ns() - _t
                    bytes_written += len(payload)
                    if bytes_written >= progress_next_threshold:
                        elapsed_ms = (time.perf_counter_ns() - fetch_started_at) // 1_000_000
                        rate_mbps = (bytes_written / max(1, elapsed_ms)) * 1000.0 / (1024 * 1024)
                        logger.info(
                            "[DIAG] fetch progress artifact_id=%s bytes=%s/%s elapsed_ms=%d rate_MBps=%.2f recv_ms=%d write_ms=%d digest_ms=%d",
                            descriptor.artifact_id,
                            bytes_written,
                            descriptor.size_bytes,
                            elapsed_ms,
                            rate_mbps,
                            recv_ns // 1_000_000,
                            write_ns // 1_000_000,
                            digest_ns // 1_000_000,
                        )
                        progress_next_threshold += progress_bucket_bytes
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
    logger.info(
        "artifact fetched artifact_id=%s bytes=%s destination=%s server_checksum=%s local_checksum=%s recv_ms=%d write_ms=%d digest_ms=%d",
        descriptor.artifact_id,
        bytes_written,
        destination_path,
        expected_checksum,
        checksum,
        recv_ns // 1_000_000,
        write_ns // 1_000_000,
        digest_ns // 1_000_000,
    )
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
    checksum = _sha256_path(local_path, chunk_size=chunk_size)
    size_bytes = local_path.stat().st_size
    logger.info(
        "artifact published artifact_id=%s bytes=%s checksum=%s producer=%s",
        artifact_id,
        size_bytes,
        checksum,
        producer_node_id,
    )
    return PublishedArtifact(
        descriptor=ArtifactDescriptor(
            artifact_id=artifact_id,
            content_type=content_type,
            size_bytes=size_bytes,
            checksum=checksum,
            producer_node_id=producer_node_id,
            transfer_host=public_host,
            transfer_port=public_port,
            chunk_size=chunk_size,
            ready=True,
        ),
        local_path=local_path,
    )
