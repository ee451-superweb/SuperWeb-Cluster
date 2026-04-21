"""Tests for the large artifact TCP data plane."""

from __future__ import annotations

import hashlib
import socket
import tempfile
import unittest
from pathlib import Path

from transport.artifact_manager import ArtifactManager
from wire.internal_protocol.data_plane_codec import (
    encode_chunk,
    encode_deliver,
    encode_download_request,
    encode_end,
)


class LargeDataTransferTests(unittest.TestCase):
    def test_publish_and_fetch_bytes_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = ArtifactManager(
                root_dir=root_dir / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = (b"superweb-artifact-" * 4096)[:131072]
                descriptor = manager.publish_bytes(
                    payload,
                    producer_node_id="main-node",
                    artifact_id="artifact-test",
                )
                fetched = manager.fetch_bytes(descriptor, timeout=5.0)
                self.assertEqual(fetched, payload)
            finally:
                manager.close()

    def test_publish_and_fetch_bytes_round_trip_with_windows_unsafe_artifact_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = ArtifactManager(
                root_dir=root_dir / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = (b"unsafe-artifact-id-" * 4096)[:65536]
                descriptor = manager.publish_bytes(
                    payload,
                    producer_node_id="compute-node",
                    artifact_id="request:1/worker?gpu*0",
                )
                fetched = manager.fetch_bytes(descriptor, timeout=5.0)
                self.assertEqual(descriptor.artifact_id, "request:1/worker?gpu*0")
                self.assertEqual(fetched, payload)
            finally:
                manager.close()


    def test_upload_slot_accepts_deliver_and_completes_future(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = ArtifactManager(
                root_dir=root_dir / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = (b"upload-payload-" * 8192)[:65536]
                checksum = hashlib.sha256(payload).hexdigest()
                future = manager.register_upload_slot(
                    upload_id="upload-test",
                    expected_size=len(payload),
                    expected_checksum=checksum,
                    expected_content_type="application/x-superweb-conv2d-weight",
                )
                with socket.create_connection(("127.0.0.1", manager.port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    sock.sendall(
                        encode_deliver(
                            upload_id="upload-test",
                            size_bytes=len(payload),
                            checksum=checksum,
                            content_type="application/x-superweb-conv2d-weight",
                        )
                    )
                    chunk_size = 16 * 1024
                    for offset in range(0, len(payload), chunk_size):
                        sock.sendall(
                            encode_chunk(offset=offset, data=payload[offset:offset + chunk_size])
                        )
                    sock.sendall(encode_end(size_bytes=len(payload)))
                destination_path = future.result(timeout=5.0)
                self.assertEqual(destination_path.read_bytes(), payload)
            finally:
                manager.close()

    def test_upload_then_download_on_same_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            manager = ArtifactManager(
                root_dir=root_dir / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                upload_payload = b"weight-bytes-" * 4096
                upload_payload = upload_payload[:32768]
                upload_checksum = hashlib.sha256(upload_payload).hexdigest()
                future = manager.register_upload_slot(
                    upload_id="upload-same-sock",
                    expected_size=len(upload_payload),
                    expected_checksum=upload_checksum,
                )
                download_payload = b"result-bytes-" * 4096
                download_payload = download_payload[:32768]
                download_descriptor = manager.publish_bytes(
                    download_payload,
                    producer_node_id="main-node",
                    artifact_id="download-same-sock",
                )

                with socket.create_connection(("127.0.0.1", manager.port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    sock.sendall(
                        encode_deliver(
                            upload_id="upload-same-sock",
                            size_bytes=len(upload_payload),
                            checksum=upload_checksum,
                            content_type="application/octet-stream",
                        )
                    )
                    chunk_size = 16 * 1024
                    for offset in range(0, len(upload_payload), chunk_size):
                        sock.sendall(
                            encode_chunk(offset=offset, data=upload_payload[offset:offset + chunk_size])
                        )
                    sock.sendall(encode_end(size_bytes=len(upload_payload)))

                    future.result(timeout=5.0)

                    sock.sendall(encode_download_request(download_descriptor.artifact_id))
                    # Drain INIT + CHUNK* + END frames.
                    from wire.internal_protocol.data_plane_codec import (
                        CHUNK_HEADER,
                        END_HEADER,
                        INIT_HEADER,
                        MSG_CHUNK,
                        MSG_END,
                        MSG_INIT,
                    )

                    def recv_exactly(n: int) -> bytes:
                        buf = bytearray()
                        while len(buf) < n:
                            chunk = sock.recv(n - len(buf))
                            if not chunk:
                                raise ConnectionError("short read")
                            buf.extend(chunk)
                        return bytes(buf)

                    head6 = recv_exactly(6)
                    self.assertEqual(head6[5], MSG_INIT)
                    init_header = head6 + recv_exactly(INIT_HEADER.size - 6)
                    _, _, _, _size, _chunk_size, checksum_len, content_type_len = INIT_HEADER.unpack(init_header)
                    recv_exactly(checksum_len + content_type_len)
                    received = bytearray()
                    while True:
                        head6 = recv_exactly(6)
                        message_type = head6[5]
                        if message_type == MSG_CHUNK:
                            chunk_header = head6 + recv_exactly(CHUNK_HEADER.size - 6)
                            _, _, _, _offset, length = CHUNK_HEADER.unpack(chunk_header)
                            received.extend(recv_exactly(length))
                        elif message_type == MSG_END:
                            end_header = head6 + recv_exactly(END_HEADER.size - 6)
                            _, _, _, total = END_HEADER.unpack(end_header)
                            self.assertEqual(total, len(received))
                            break
                        else:
                            self.fail(f"unexpected message_type={message_type}")
                    self.assertEqual(bytes(received), download_payload)
            finally:
                manager.close()


class TransferTimingInstrumentationTests(unittest.TestCase):
    """Use this to pin the per-stage timing fields in transport log lines.

    The fields let operators split the fetch/upload wall clock across
    recv (pure socket), write (disk) and digest (SHA256) so that gaps
    between business-level throughput and raw iperf3 numbers are
    attributable.
    """

    def test_fetch_log_reports_recv_write_digest_ms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(
                root_dir=Path(temp_dir) / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = (b"timing-" * 8192)[:65536]
                descriptor = manager.publish_bytes(
                    payload,
                    producer_node_id="main-node",
                    artifact_id="timing-fetch",
                )
                with self.assertLogs("transport.large_data_transfer", level="INFO") as cm:
                    manager.fetch_bytes(descriptor, timeout=5.0)
            finally:
                manager.close()
            fetched_lines = [line for line in cm.output if "artifact fetched" in line]
            self.assertTrue(fetched_lines, "expected at least one 'artifact fetched' log line")
            self.assertIn("recv_ms=", fetched_lines[-1])
            self.assertIn("write_ms=", fetched_lines[-1])
            self.assertIn("digest_ms=", fetched_lines[-1])

    def test_upload_log_reports_recv_write_digest_ms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(
                root_dir=Path(temp_dir) / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = (b"upload-timing-" * 4096)[:32768]
                checksum = hashlib.sha256(payload).hexdigest()
                future = manager.register_upload_slot(
                    upload_id="timing-upload",
                    expected_size=len(payload),
                    expected_checksum=checksum,
                )
                with self.assertLogs("transport.large_data_transfer", level="INFO") as cm:
                    with socket.create_connection(("127.0.0.1", manager.port), timeout=5.0) as sock:
                        sock.settimeout(5.0)
                        sock.sendall(
                            encode_deliver(
                                upload_id="timing-upload",
                                size_bytes=len(payload),
                                checksum=checksum,
                                content_type="application/octet-stream",
                            )
                        )
                        chunk_size = 16 * 1024
                        for offset in range(0, len(payload), chunk_size):
                            sock.sendall(
                                encode_chunk(offset=offset, data=payload[offset:offset + chunk_size])
                            )
                        sock.sendall(encode_end(size_bytes=len(payload)))
                    future.result(timeout=5.0)
            finally:
                manager.close()
            accepted_lines = [line for line in cm.output if "upload accepted" in line]
            self.assertTrue(accepted_lines, "expected at least one 'upload accepted' log line")
            self.assertIn("recv_ms=", accepted_lines[-1])
            self.assertIn("write_ms=", accepted_lines[-1])
            self.assertIn("digest_ms=", accepted_lines[-1])


if __name__ == "__main__":
    unittest.main()
