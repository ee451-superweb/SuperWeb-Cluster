"""Dispatcher tests focused on GEMM M-axis assignment."""

from __future__ import annotations

import unittest
from unittest import mock

from core.constants import METHOD_GEMM
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
        method=METHOD_GEMM,
        hardware_type="cuda",
        effective_gflops=gflops,
        rank=1,
    )


class DispatcherGemmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = TaskDispatcher()

    def test_dispatch_gemm_returns_one_assignment_per_schedulable_worker(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_gemm(
            request_id="req-1",
            rows=1024,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=100.0),
                _make_capability(peer_id="peer-b", runtime_id="worker-2", gflops=100.0),
            ],
        )

        self.assertEqual(len(assignments), 2)
        self.assertTrue(all(isinstance(a, WorkerTaskSlice) for a in assignments))
        self.assertTrue(all(a.method == METHOD_GEMM for a in assignments))
        self.assertEqual(assignments[0].m_start, 0)
        self.assertEqual(assignments[0].m_end, 512)
        self.assertEqual(assignments[1].m_start, 512)
        self.assertEqual(assignments[1].m_end, 1024)
        self.assertTrue(all(a.row_start == 0 and a.row_end == 0 for a in assignments))
        self.assertTrue(all(a.start_oc == 0 and a.end_oc == 0 for a in assignments))

    def test_dispatch_gemm_partitions_m_axis_by_gflops_weight(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_gemm(
            request_id="req-2",
            rows=4096,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=300.0),
                _make_capability(peer_id="peer-b", runtime_id="worker-2", gflops=100.0),
            ],
        )

        self.assertEqual(len(assignments), 2)
        span_a = assignments[0].m_end - assignments[0].m_start
        span_b = assignments[1].m_end - assignments[1].m_start
        self.assertGreater(span_a, span_b)
        self.assertEqual(span_a + span_b, 4096)
        # The two slices should tile the M range contiguously.
        self.assertEqual(assignments[0].m_end, assignments[1].m_start)

    def test_dispatch_gemm_skips_workers_with_zero_gflops(self) -> None:
        worker_a = _make_worker("peer-a", "worker-1")
        worker_b = _make_worker("peer-b", "worker-2")
        assignments = self.dispatcher.dispatch_gemm(
            request_id="req-3",
            rows=2048,
            workers=[worker_a, worker_b],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-1", gflops=50.0),
            ],
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].connection.peer_id, "peer-a")
        self.assertEqual(assignments[0].m_start, 0)
        self.assertEqual(assignments[0].m_end, 2048)

    def test_dispatch_gemm_returns_empty_when_no_cuda_workers(self) -> None:
        """On a CUDA-less cluster nothing advertises GEMM capacity."""
        worker_a = _make_worker("peer-a", "worker-1")
        assignments = self.dispatcher.dispatch_gemm(
            request_id="req-4",
            rows=1024,
            workers=[worker_a],
            worker_hardware=[],
        )

        self.assertEqual(assignments, [])

    def test_dispatch_gemm_artifact_id_encodes_runtime(self) -> None:
        worker_a = _make_worker("peer-a", "worker-7")
        assignments = self.dispatcher.dispatch_gemm(
            request_id="req-5",
            rows=512,
            workers=[worker_a],
            worker_hardware=[
                _make_capability(peer_id="peer-a", runtime_id="worker-7", gflops=10.0),
            ],
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].artifact_id, "req-5:worker-7")
        self.assertEqual(assignments[0].task_id, "req-5")


if __name__ == "__main__":
    unittest.main()
