"""Weight-slice behavior of the main-node task exchange."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.config import AppConfig
from main_node.task_exchange import WorkerTaskExchange


def _make_spec(*, k: int, c_in: int) -> SimpleNamespace:
    return SimpleNamespace(k=k, c_in=c_in)


def _pack_floats(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


class Conv2dWeightSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.dataset_dir = Path(self._tempdir.name)
        self.exchange = WorkerTaskExchange(
            config=AppConfig(),
            conv2d_dataset_dir=self.dataset_dir,
            remove_worker_connection=mock.Mock(),
            artifact_manager=None,
            logger=mock.Mock(),
        )

    def test_returns_contiguous_channel_slice_from_explicit_weight_path(self) -> None:
        spec = _make_spec(k=2, c_in=1)
        channel_0 = _pack_floats([1.0, 2.0, 3.0, 4.0])
        channel_1 = _pack_floats([5.0, 6.0, 7.0, 8.0])
        channel_2 = _pack_floats([9.0, 10.0, 11.0, 12.0])
        weight_path = self.dataset_dir / "custom_weight.bin"
        weight_path.write_bytes(channel_0 + channel_1 + channel_2)

        first_two = self.exchange.conv2d_weight_slice(
            variant="small",
            spec=spec,
            start_oc=0,
            end_oc=2,
            weight_path=weight_path,
        )
        middle_one = self.exchange.conv2d_weight_slice(
            variant="small",
            spec=spec,
            start_oc=1,
            end_oc=2,
            weight_path=weight_path,
        )

        self.assertEqual(first_two, channel_0 + channel_1)
        self.assertEqual(middle_one, channel_1)

    def test_uses_variant_filename_when_weight_path_is_none(self) -> None:
        spec = _make_spec(k=1, c_in=1)
        channel_payload = _pack_floats([42.0])
        (self.dataset_dir / "small_weight.bin").write_bytes(channel_payload * 2)

        slice_bytes = self.exchange.conv2d_weight_slice(
            variant="small",
            spec=spec,
            start_oc=0,
            end_oc=1,
        )

        self.assertEqual(slice_bytes, channel_payload)

    def test_falls_back_to_legacy_variant_filename(self) -> None:
        spec = _make_spec(k=1, c_in=1)
        channel_payload = _pack_floats([3.5])
        (self.dataset_dir / "test_weight.bin").write_bytes(channel_payload * 2)

        slice_bytes = self.exchange.conv2d_weight_slice(
            variant="small",
            spec=spec,
            start_oc=1,
            end_oc=2,
        )

        self.assertEqual(slice_bytes, channel_payload)

    def test_raises_when_weight_file_is_missing(self) -> None:
        spec = _make_spec(k=1, c_in=1)
        with self.assertRaises(FileNotFoundError):
            self.exchange.conv2d_weight_slice(
                variant="small",
                spec=spec,
                start_oc=0,
                end_oc=1,
                weight_path=self.dataset_dir / "does-not-exist.bin",
            )


class ClientWeightPathRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.exchange = WorkerTaskExchange(
            config=AppConfig(),
            conv2d_dataset_dir=Path(self._tempdir.name),
            remove_worker_connection=mock.Mock(),
            logger=mock.Mock(),
        )

    def test_register_and_unregister_round_trip(self) -> None:
        weight_path = Path(self._tempdir.name) / "client-weight.bin"
        self.exchange.register_client_weight_path("req-1", weight_path)

        popped = self.exchange.unregister_client_weight_path("req-1")

        self.assertEqual(popped, weight_path)
        self.assertIsNone(self.exchange.unregister_client_weight_path("req-1"))

    def test_unregister_missing_request_id_returns_none(self) -> None:
        self.assertIsNone(self.exchange.unregister_client_weight_path("never-seen"))


if __name__ == "__main__":
    unittest.main()
