"""Tests for the large artifact TCP data plane."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from transport.artifact_manager import ArtifactManager


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


if __name__ == "__main__":
    unittest.main()
