"""Dispatcher tests focused on conv2d output-channel assignment."""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from core.constants import METHOD_CONV2D
from main_node.dispatcher import TaskDispatcher, WorkerTaskSlice
from main_node.registry import RuntimePeerConnection, WorkerHardwareCapability


def _make_worker(peer_id: str, runtime_id: str) -> RuntimePeerConnection:
    return RuntimePeerConnection(
        peer_id=peer_id,
        runtime_id=runtime_id,
        node_name=f"worker-{runtime_id}",
        role="worker",
        peer_address="10.0.0.2",
        peer_port=5000,
        sock=mock.Mock(),
    )


def _make_capability(
    *, peer_id: str, runtime_id: str, gflops: float
) -> WorkerHardwareCapability:
    return WorkerHardwareCapability(
        hardware_id=f"{peer_id}:cuda",
        worker_peer_id=peer_id,
        worker_runtime_id=runtime_id,
        worker_node_name=f"worker-{runtime_id}",
        method=METHOD_CONV2D,
        hardware_type="cuda",
        effective_gflops=gflops,
        rank=1,
    )


class DispatcherConv2dTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = TaskDispatcher()

    def test_dispatch_conv2d_returns_one_assignment_per_schedulable_worker(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_conv2d(
            request_id="req-1",
            output_channels=32,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=100.0),
                _make_capability(peer_id="peer-b", runtime_id="worker-2", gflops=100.0),
            ],
        )

        self.assertEqual(len(assignments), 2)
        self.assertTrue(all(isinstance(a, WorkerTaskSlice) for a in assignments))
        self.assertTrue(all(a.method == METHOD_CONV2D for a in assignments))
        self.assertEqual(assignments[0].start_oc, 0)
        self.assertEqual(assignments[0].end_oc, 16)
        self.assertEqual(assignments[1].start_oc, 16)
        self.assertEqual(assignments[1].end_oc, 32)

    def test_dispatch_conv2d_partitions_channels_by_gflops_weight(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_conv2d(
            request_id="req-2",
            output_channels=40,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=30.0),
                _make_capability(peer_id="peer-b", runtime_id="worker-2", gflops=10.0),
            ],
        )

        self.assertEqual(len(assignments), 2)
        span_a = assignments[0].end_oc - assignments[0].start_oc
        span_b = assignments[1].end_oc - assignments[1].start_oc
        self.assertGreater(span_a, span_b)
        self.assertEqual(span_a + span_b, 40)

    def test_dispatch_conv2d_skips_workers_with_zero_gflops(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_conv2d(
            request_id="req-3",
            output_channels=24,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=50.0),
            ],
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].connection.peer_id, "peer-a")
        self.assertEqual(assignments[0].start_oc, 0)
        self.assertEqual(assignments[0].end_oc, 24)

    def test_dispatch_conv2d_returns_empty_when_no_schedulable_workers(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        assignments = self.dispatcher.dispatch_conv2d(
            request_id="req-4",
            output_channels=16,
            workers=[worker_a],
            worker_hardware=[],
        )

        self.assertEqual(assignments, [])

    def test_dispatch_conv2d_artifact_id_encodes_runtime_and_zero_chunk(self) -> None:
        worker_a = _make_worker("peer-a", "worker-7")
        assignments = self.dispatcher.dispatch_conv2d(
            request_id="req-5",
            output_channels=8,
            workers=[worker_a],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-7", gflops=10.0),
            ],
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].artifact_id, "req-5:worker-7:0")
        self.assertEqual(assignments[0].task_id, "req-5")


if __name__ == "__main__":
    unittest.main()
