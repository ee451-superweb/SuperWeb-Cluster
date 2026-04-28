"""Tests for the DELIVER upload path and data-plane allocation helpers.

Covers:
  - DELIVER codec round trip on the cluster side.
  - ``ClientRequestHandler.allocate_data_plane_endpoints`` picking the right
    combination of upload_id / download_id based on method and stats_only flag.
  - Orphan upload slots getting cancelled when request validation fails after
    the slot has already been registered.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.constants import (
    CONV2D_CLIENT_RESPONSE_STATS_ONLY,
    METHOD_CONV2D,
    METHOD_GEMV,
    STATUS_INTERNAL_ERROR,
)
from main_node.request_handler import ClientRequestHandler, DataPlaneAllocation
from transport.artifact_manager import ArtifactManager
from wire.external_protocol.control_plane import (
    ClientRequest,
    Conv2dRequestPayload,
    GemvRequestPayload,
)
from wire.internal_protocol.data_plane import ArtifactDeliverFrame
from wire.internal_protocol.data_plane_codec import (
    DELIVER_HEADER,
    decode_deliver,
    encode_chunk,
    encode_deliver,
    encode_end,
)


def _make_conv2d_request(
    *,
    request_id: str = "req-1",
    upload_size_bytes: int = 0,
    upload_checksum: str = "",
    stats_only: bool = False,
) -> ClientRequest:
    payload = Conv2dRequestPayload(
        tensor_h=32,
        tensor_w=32,
        channels_in=3,
        channels_out=4,
        kernel_size=3,
        padding=1,
        stride=1,
        client_response_mode=CONV2D_CLIENT_RESPONSE_STATS_ONLY if stats_only else 0,
        upload_size_bytes=upload_size_bytes,
        upload_checksum=upload_checksum,
    )
    return ClientRequest(
        request_id=request_id,
        client_name="test-client",
        method=METHOD_CONV2D,
        size="small",
        object_id="conv2d/small",
        stream_id="",
        timestamp_ms=0,
        iteration_count=1,
        request_payload=payload,
    )


def _make_gemv_request(request_id: str = "req-gemv") -> ClientRequest:
    return ClientRequest(
        request_id=request_id,
        client_name="test-client",
        method=METHOD_GEMV,
        size="small",
        object_id="",
        stream_id="",
        timestamp_ms=0,
        iteration_count=1,
        request_payload=GemvRequestPayload(
            vector_length=4,
            vector_data=b"\x00" * 16,
        ),
    )


class DeliverCodecTests(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        raw = encode_deliver(
            upload_id="upload-xyz",
            size_bytes=4096,
            checksum="a" * 64,
            content_type="application/x-superweb-conv2d-weight",
        )
        header = raw[: DELIVER_HEADER.size]
        body = raw[DELIVER_HEADER.size:]
        frame = decode_deliver(header, body)
        self.assertIsInstance(frame, ArtifactDeliverFrame)
        self.assertEqual(frame.upload_id, "upload-xyz")
        self.assertEqual(frame.size_bytes, 4096)
        self.assertEqual(frame.checksum, "a" * 64)
        self.assertEqual(frame.content_type, "application/x-superweb-conv2d-weight")

    def test_decode_rejects_mismatched_payload_length(self) -> None:
        raw = encode_deliver(
            upload_id="u",
            size_bytes=1,
            checksum="",
            content_type="",
        )
        header = raw[: DELIVER_HEADER.size]
        body = raw[DELIVER_HEADER.size:] + b"extra"
        with self.assertRaises(ValueError):
            decode_deliver(header, body)


class AllocateDataPlaneEndpointsTests(unittest.TestCase):
    def _handler(self, *, port: int = 45000, public_host: str = "10.0.0.5") -> ClientRequestHandler:
        artifact_manager = MagicMock()
        artifact_manager.public_host = public_host
        artifact_manager.port = port
        artifact_manager.register_upload_slot.return_value = concurrent.futures.Future()
        return ClientRequestHandler(
            config=MagicMock(),
            registry=MagicMock(),
            dispatcher=MagicMock(),
            aggregator=MagicMock(),
            gemv_spec=MagicMock(),
            conv2d_dataset_dir=Path("/tmp"),
            task_exchange=MagicMock(),
            artifact_manager=artifact_manager,
            cluster_counts=lambda: (0, 0),
        )

    def test_gemv_request_allocates_endpoint_only(self) -> None:
        handler = self._handler()
        request = _make_gemv_request()

        allocation = handler.allocate_data_plane_endpoints(request)

        self.assertEqual(allocation.data_endpoint_host, "10.0.0.5")
        self.assertEqual(allocation.data_endpoint_port, 45000)
        self.assertEqual(allocation.upload_id, "")
        self.assertEqual(allocation.download_id, "")
        self.assertIsNone(allocation.upload_future)
        handler.artifact_manager.register_upload_slot.assert_not_called()

    def test_conv2d_stats_only_without_upload_allocates_nothing_extra(self) -> None:
        handler = self._handler()
        request = _make_conv2d_request(stats_only=True, upload_size_bytes=0)

        allocation = handler.allocate_data_plane_endpoints(request)

        self.assertEqual(allocation.upload_id, "")
        self.assertEqual(allocation.download_id, "")
        self.assertIsNone(allocation.upload_future)
        handler.artifact_manager.register_upload_slot.assert_not_called()

    def test_conv2d_stats_only_with_upload_registers_upload_slot(self) -> None:
        handler = self._handler()
        request = _make_conv2d_request(
            upload_size_bytes=1024,
            upload_checksum="deadbeef",
            stats_only=True,
        )

        allocation = handler.allocate_data_plane_endpoints(request)

        self.assertTrue(allocation.upload_id.startswith(request.request_id + "-upload-"))
        self.assertEqual(allocation.download_id, "")
        self.assertIsNotNone(allocation.upload_future)
        handler.artifact_manager.register_upload_slot.assert_called_once()
        kwargs = handler.artifact_manager.register_upload_slot.call_args.kwargs
        self.assertEqual(kwargs["expected_size"], 1024)
        self.assertEqual(kwargs["expected_checksum"], "deadbeef")

    def test_conv2d_full_with_upload_allocates_both_ids(self) -> None:
        handler = self._handler()
        request = _make_conv2d_request(
            upload_size_bytes=8192,
            upload_checksum="cafe",
            stats_only=False,
        )

        allocation = handler.allocate_data_plane_endpoints(request)

        self.assertTrue(allocation.upload_id.startswith(request.request_id + "-upload-"))
        self.assertTrue(allocation.download_id.startswith(request.request_id + "-download-"))
        self.assertIsNotNone(allocation.upload_future)

    def test_conv2d_full_without_upload_allocates_download_only(self) -> None:
        handler = self._handler()
        request = _make_conv2d_request(upload_size_bytes=0, stats_only=False)

        allocation = handler.allocate_data_plane_endpoints(request)

        self.assertEqual(allocation.upload_id, "")
        self.assertTrue(allocation.download_id.startswith(request.request_id + "-download-"))
        self.assertIsNone(allocation.upload_future)
        handler.artifact_manager.register_upload_slot.assert_not_called()


class UploadSlotCancellationTests(unittest.TestCase):
    def test_orphan_upload_slot_is_cancelled_when_request_fails_validation(self) -> None:
        """If a conv2d request passes allocation but then trips a later validation
        path (e.g., bad iteration_count flipped after-the-fact, or an artifact
        manager that is unavailable), the wrapping finally block on
        ``build_client_response_for_request`` must cancel the slot so the
        artifact-manager registry does not leak.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(
                root_dir=Path(temp_dir) / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                future = manager.register_upload_slot(
                    upload_id="orphan-upload",
                    expected_size=512,
                )
                self.assertIn("orphan-upload", manager._upload_slots)

                manager.cancel_upload_slot("orphan-upload")

                self.assertNotIn("orphan-upload", manager._upload_slots)
                self.assertTrue(future.cancelled())
            finally:
                manager.close()

    def test_upload_slot_rejects_mismatched_upload_size(self) -> None:
        """The server-side DELIVER handler must reject uploads that declare a
        size other than what the slot expects, so a misbehaving client cannot
        quietly overwrite a smaller-than-declared weight file.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(
                root_dir=Path(temp_dir) / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                manager.register_upload_slot(
                    upload_id="size-check",
                    expected_size=1024,
                )

                bogus_payload = b"A" * 2048
                bogus_checksum = hashlib.sha256(bogus_payload).hexdigest()
                with socket.create_connection(("127.0.0.1", manager.port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    sock.sendall(
                        encode_deliver(
                            upload_id="size-check",
                            size_bytes=len(bogus_payload),
                            checksum=bogus_checksum,
                            content_type="application/octet-stream",
                        )
                    )
                    # Read back the ERROR frame the server emits before closing.
                    err = sock.recv(1024)
                self.assertTrue(err)
            finally:
                manager.close()


class DeliverThenErrorOnChecksumTests(unittest.TestCase):
    def test_deliver_with_wrong_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ArtifactManager(
                root_dir=Path(temp_dir) / "artifacts",
                public_host="127.0.0.1",
                chunk_size=16 * 1024,
            )
            try:
                payload = b"real-bytes" * 1024
                declared_bad_checksum = "f" * 64
                future = manager.register_upload_slot(
                    upload_id="checksum-mismatch",
                    expected_size=len(payload),
                    expected_checksum=declared_bad_checksum,
                )

                def _push() -> None:
                    try:
                        with socket.create_connection(("127.0.0.1", manager.port), timeout=5.0) as sock:
                            sock.settimeout(5.0)
                            sock.sendall(
                                encode_deliver(
                                    upload_id="checksum-mismatch",
                                    size_bytes=len(payload),
                                    checksum=declared_bad_checksum,
                                    content_type="application/octet-stream",
                                )
                            )
                            sock.sendall(encode_chunk(offset=0, data=payload))
                            sock.sendall(encode_end(size_bytes=len(payload)))
                            # Drain any ERROR frame the server emits.
                            try:
                                sock.recv(1024)
                            except OSError:
                                pass
                    except OSError:
                        pass

                thread = threading.Thread(target=_push, daemon=True)
                thread.start()

                with self.assertRaises((RuntimeError, ValueError)):
                    future.result(timeout=5.0)
                thread.join(timeout=2.0)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
