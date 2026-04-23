"""Tests for worker-level task failover in ClientRequestHandler."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from core.constants import METHOD_CONV2D, METHOD_GEMV
from main_node.dispatcher import WorkerTaskSlice
from main_node.request_handler import ClientRequestHandler
from wire.internal_protocol.transport import TaskResult, WorkerTiming


def _make_handler(task_exchange) -> ClientRequestHandler:
    return ClientRequestHandler(
        config=MagicMock(),
        registry=MagicMock(),
        dispatcher=MagicMock(),
        aggregator=MagicMock(),
        gemv_spec=MagicMock(),
        conv2d_dataset_dir=Path("."),
        task_exchange=task_exchange,
        artifact_manager=MagicMock(),
        cluster_counts=lambda: (0, 0),
    )


def _make_worker_connection(runtime_id: str) -> mock.Mock:
    connection = mock.Mock()
    connection.runtime_id = runtime_id
    connection.peer_id = f"worker:{runtime_id}@10.0.0.1:5000"
    connection.node_name = runtime_id
    connection.io_lock = threading.Lock()
    connection.task_lock = threading.Lock()
    return connection


def _gemv_slice(connection, *, task_id: str, row_start: int, row_end: int, gflops: float) -> WorkerTaskSlice:
    return WorkerTaskSlice(
        connection=connection,
        task_id=task_id,
        artifact_id=f"{task_id}:{connection.runtime_id}",
        row_start=row_start,
        row_end=row_end,
        effective_gflops=gflops,
        method=METHOD_GEMV,
    )


def _gemv_result(assignment: WorkerTaskSlice) -> TaskResult:
    length = assignment.row_end - assignment.row_start
    return TaskResult(
        request_id="req-1",
        node_id=assignment.connection.runtime_id,
        task_id=assignment.task_id,
        timestamp_ms=0,
        status_code=200,
        row_start=assignment.row_start,
        row_end=assignment.row_end,
        output_length=length,
        output_vector=b"\x00" * (length * 4),
        iteration_count=1,
    )


def _gemv_timing(assignment: WorkerTaskSlice) -> WorkerTiming:
    return WorkerTiming(
        node_id=assignment.connection.runtime_id,
        task_id=assignment.task_id,
        slice=f"rows={assignment.row_start}:{assignment.row_end}",
        wall_ms=1,
        artifact_fetch_ms=0,
    )


class FailoverTests(unittest.TestCase):
    """Validate worker-level task re-dispatch when a slice raises."""

    def test_happy_path_returns_every_outcome(self) -> None:
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        assignments = [
            _gemv_slice(worker_a, task_id="req-1", row_start=0, row_end=50, gflops=100.0),
            _gemv_slice(worker_b, task_id="req-1", row_start=50, row_end=100, gflops=100.0),
        ]
        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = lambda req, a: (
            _gemv_result(a),
            _gemv_timing(a),
        )
        handler = _make_handler(task_exchange)
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV)

        outcomes = handler._run_assignments_with_failover(request, assignments)

        self.assertEqual(len(outcomes), 2)
        covered = sorted(
            (result.row_start, result.row_end) for result, _ in outcomes
        )
        self.assertEqual(covered, [(0, 50), (50, 100)])

    def test_single_worker_failure_redispatches_range_to_survivors(self) -> None:
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        worker_c = _make_worker_connection("worker-c")
        assignments = [
            _gemv_slice(worker_a, task_id="req-1", row_start=0, row_end=30, gflops=100.0),
            _gemv_slice(worker_b, task_id="req-1", row_start=30, row_end=60, gflops=100.0),
            _gemv_slice(worker_c, task_id="req-1", row_start=60, row_end=100, gflops=100.0),
        ]

        def run_slice(_request, assignment):
            if assignment.connection.runtime_id == "worker-a" and assignment.task_id == "req-1":
                raise ConnectionError("worker-a died mid-task")
            return _gemv_result(assignment), _gemv_timing(assignment)

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = run_slice
        handler = _make_handler(task_exchange)
        handler.logger = mock.Mock()
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV)

        outcomes = handler._run_assignments_with_failover(request, assignments)

        # Original two survivors (b, c) plus two retry slices covering worker-a's range.
        self.assertEqual(len(outcomes), 4)
        covered = sorted((result.row_start, result.row_end) for result, _ in outcomes)
        # Original range for worker-a [0, 30) must be split between b and c.
        self.assertEqual(covered[0], (0, 15))
        self.assertEqual(covered[1], (15, 30))
        self.assertEqual(covered[2], (30, 60))
        self.assertEqual(covered[3], (60, 100))
        # Retries must not target the dead worker.
        dispatched_runtime_ids = [
            call.args[1].connection.runtime_id
            for call in task_exchange.run_worker_task_slice.call_args_list
        ]
        retry_targets = [rid for rid in dispatched_runtime_ids if rid != "worker-a"]
        self.assertNotIn("worker-a", retry_targets[2:])

    def test_all_workers_fail_raises_runtime_error(self) -> None:
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        assignments = [
            _gemv_slice(worker_a, task_id="req-1", row_start=0, row_end=50, gflops=100.0),
            _gemv_slice(worker_b, task_id="req-1", row_start=50, row_end=100, gflops=100.0),
        ]
        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = ConnectionError("all workers dead")
        handler = _make_handler(task_exchange)
        handler.logger = mock.Mock()
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV)

        with self.assertRaises(RuntimeError) as ctx:
            handler._run_assignments_with_failover(request, assignments)
        self.assertIn("all workers failed", str(ctx.exception))

    def test_retry_slice_has_unique_task_and_artifact_ids(self) -> None:
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        failed_slice = _gemv_slice(
            worker_a, task_id="req-1", row_start=0, row_end=40, gflops=100.0,
        )
        handler = _make_handler(MagicMock())
        retry_counter = [0]
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV)

        retry_slices = handler._build_retry_assignments(
            request,
            failed_slice,
            [worker_b],
            {worker_a.peer_id: 100.0, worker_b.peer_id: 100.0},
            retry_counter,
        )

        self.assertEqual(len(retry_slices), 1)
        retry = retry_slices[0]
        self.assertEqual(retry.task_id, "req-1-r0")
        self.assertEqual(retry.artifact_id, "req-1-r0:worker-b")
        self.assertEqual((retry.row_start, retry.row_end), (0, 40))
        self.assertEqual(retry.connection.runtime_id, "worker-b")
        self.assertEqual(retry_counter[0], 1)

    def test_build_retry_assignments_covers_conv2d_channels_contiguously(self) -> None:
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        worker_c = _make_worker_connection("worker-c")
        failed_slice = WorkerTaskSlice(
            connection=worker_a,
            task_id="req-1",
            artifact_id="req-1:worker-a:0",
            row_start=0,
            row_end=0,
            effective_gflops=100.0,
            method=METHOD_CONV2D,
            start_oc=0,
            end_oc=32,
        )
        handler = _make_handler(MagicMock())
        retry_counter = [5]
        request = mock.Mock(request_id="req-1", method=METHOD_CONV2D)

        retry_slices = handler._build_retry_assignments(
            request,
            failed_slice,
            [worker_b, worker_c],
            {
                worker_a.peer_id: 100.0,
                worker_b.peer_id: 100.0,
                worker_c.peer_id: 100.0,
            },
            retry_counter,
        )

        self.assertEqual(len(retry_slices), 2)
        self.assertEqual(retry_slices[0].method, METHOD_CONV2D)
        self.assertEqual(retry_slices[0].task_id, "req-1-r5")
        self.assertEqual(retry_slices[0].artifact_id, "req-1-r5:worker-b:0")
        ranges = [(slice_.start_oc, slice_.end_oc) for slice_ in retry_slices]
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 32)
        # Contiguity check.
        for left, right in zip(ranges, ranges[1:]):
            self.assertEqual(left[1], right[0])


if __name__ == "__main__":
    unittest.main()
