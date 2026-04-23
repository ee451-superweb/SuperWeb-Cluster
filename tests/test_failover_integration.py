"""Multi-worker failover integration tests with real thread concurrency.

These cases differ from ``test_failover.py`` (which uses instant MagicMock
side-effects) by giving each simulated worker a dedicated ``threading.Event``
so the test can orchestrate the wall-clock relationship between workers:
start several slices, let one crash mid-task while the others are still
blocked, then verify failover redispatch covers the crashed worker's range
and aggregation succeeds.

The intent is to mirror the operator-facing "open several shells, kill one
worker in the middle of a gemv --iteration run" scenario without spinning
up real bootstrap.py processes.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from core.constants import METHOD_CONV2D, METHOD_GEMV
from main_node.dispatcher import WorkerTaskSlice
from main_node.request_handler import ClientRequestHandler
from wire.internal_protocol.transport import TaskResult, WorkerTiming


def _make_worker_connection(runtime_id: str) -> mock.Mock:
    connection = mock.Mock()
    connection.runtime_id = runtime_id
    connection.peer_id = f"worker:{runtime_id}@10.0.0.1:5000"
    connection.node_name = runtime_id
    connection.io_lock = threading.Lock()
    connection.task_lock = threading.Lock()
    return connection


def _gemv_slice(
    connection,
    *,
    task_id: str,
    row_start: int,
    row_end: int,
    gflops: float = 100.0,
) -> WorkerTaskSlice:
    return WorkerTaskSlice(
        connection=connection,
        task_id=task_id,
        artifact_id=f"{task_id}:{connection.runtime_id}",
        row_start=row_start,
        row_end=row_end,
        effective_gflops=gflops,
        method=METHOD_GEMV,
    )


def _conv2d_slice(
    connection,
    *,
    task_id: str,
    start_oc: int,
    end_oc: int,
    gflops: float = 100.0,
) -> WorkerTaskSlice:
    return WorkerTaskSlice(
        connection=connection,
        task_id=task_id,
        artifact_id=f"{task_id}:{connection.runtime_id}:0",
        row_start=0,
        row_end=0,
        effective_gflops=gflops,
        method=METHOD_CONV2D,
        start_oc=start_oc,
        end_oc=end_oc,
    )


def _gemv_result(assignment: WorkerTaskSlice, iteration_count: int) -> TaskResult:
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
        iteration_count=iteration_count,
    )


def _conv2d_result(assignment: WorkerTaskSlice, iteration_count: int) -> TaskResult:
    return TaskResult(
        request_id="req-1",
        node_id=assignment.connection.runtime_id,
        task_id=assignment.task_id,
        timestamp_ms=0,
        status_code=200,
        row_start=0,
        row_end=0,
        start_oc=assignment.start_oc,
        end_oc=assignment.end_oc,
        output_length=assignment.end_oc - assignment.start_oc,
        output_vector=b"\x00" * ((assignment.end_oc - assignment.start_oc) * 4),
        iteration_count=iteration_count,
    )


def _timing(assignment: WorkerTaskSlice) -> WorkerTiming:
    scope = (
        f"rows={assignment.row_start}:{assignment.row_end}"
        if assignment.method == METHOD_GEMV
        else f"oc={assignment.start_oc}:{assignment.end_oc}"
    )
    return WorkerTiming(
        node_id=assignment.connection.runtime_id,
        task_id=assignment.task_id,
        slice=scope,
        wall_ms=1,
        artifact_fetch_ms=0,
    )


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
        logger=mock.Mock(),
    )


class _WorkerSimulator:
    """Tiny stand-in that lets tests block, release, or crash each worker slice.

    Each simulated worker owns a ``proceed`` event (so the test can hold the
    slice mid-flight to mirror a running computation) and an optional
    ``exception`` override (raised instead of returning a result when the
    event fires).
    """

    def __init__(self, iteration_count: int = 1) -> None:
        self._iteration_count = iteration_count
        self._lock = threading.Lock()
        self._proceed_events: dict[str, threading.Event] = {}
        self._exceptions: dict[str, Exception] = {}
        self._call_order: list[str] = []
        self.captured_iteration_counts: list[int] = []

    def configure(
        self,
        runtime_id: str,
        *,
        proceed: threading.Event | None = None,
        exception: Exception | None = None,
    ) -> threading.Event:
        """Register a worker's blocking event and optional crash-on-release."""
        event = proceed if proceed is not None else threading.Event()
        with self._lock:
            self._proceed_events[runtime_id] = event
            if exception is not None:
                self._exceptions[runtime_id] = exception
        return event

    def run(self, request, assignment: WorkerTaskSlice):
        runtime_id = assignment.connection.runtime_id
        with self._lock:
            self._call_order.append(runtime_id)
            event = self._proceed_events.get(runtime_id)
            exc = self._exceptions.get(runtime_id)
        if event is not None:
            event.wait(timeout=5.0)
        self.captured_iteration_counts.append(request.iteration_count)
        if exc is not None:
            raise exc
        if assignment.method == METHOD_GEMV:
            return _gemv_result(assignment, request.iteration_count), _timing(assignment)
        return _conv2d_result(assignment, request.iteration_count), _timing(assignment)

    @property
    def call_order(self) -> list[str]:
        with self._lock:
            return list(self._call_order)


class MultiWorkerFailoverIntegrationTests(unittest.TestCase):
    """Higher-fidelity scheduling checks with real concurrent threads."""

    def test_worker_dies_mid_task_with_others_still_running(self) -> None:
        """Worker-a crashes after workers b/c are already in-flight.

        Mirrors the "three shells running, kill shell 1" scenario — when worker-a
        raises, its range must be redispatched across the still-running b and c
        without cancelling their in-flight work.
        """
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        worker_c = _make_worker_connection("worker-c")
        assignments = [
            _gemv_slice(worker_a, task_id="req-1", row_start=0, row_end=30),
            _gemv_slice(worker_b, task_id="req-1", row_start=30, row_end=60),
            _gemv_slice(worker_c, task_id="req-1", row_start=60, row_end=100),
        ]
        simulator = _WorkerSimulator()
        event_a = simulator.configure(
            "worker-a", exception=ConnectionError("worker-a shell killed mid-task"),
        )
        event_b = simulator.configure("worker-b")
        event_c = simulator.configure("worker-c")

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = simulator.run
        handler = _make_handler(task_exchange)
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV, iteration_count=5)

        result_holder: dict = {}

        def _invoke() -> None:
            result_holder["outcomes"] = handler._run_assignments_with_failover(
                request, assignments,
            )

        runner = threading.Thread(target=_invoke)
        runner.start()

        # Let the original three slices actually enter the fake run() body.
        deadline = time.monotonic() + 2.0
        while len(simulator.call_order) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(set(simulator.call_order), {"worker-a", "worker-b", "worker-c"})

        # Kill worker-a first; workers b and c are still blocked on their events,
        # so the failover scheduler must redispatch worker-a's range to them
        # while they are still running their original slices.
        event_a.set()
        event_b.set()
        event_c.set()

        runner.join(timeout=5.0)
        self.assertFalse(runner.is_alive(), "failover runner did not finish in time")
        outcomes = result_holder["outcomes"]

        # Original survivors (b, c) plus two retry slices covering worker-a's 30 rows.
        self.assertEqual(len(outcomes), 4)
        covered = sorted((result.row_start, result.row_end) for result, _ in outcomes)
        self.assertEqual(
            covered,
            [(0, 15), (15, 30), (30, 60), (60, 100)],
        )
        # Retry dispatch must not have ever targeted worker-a.
        dispatched = [
            call.args[1].connection.runtime_id
            for call in task_exchange.run_worker_task_slice.call_args_list
        ]
        # First three are the original fan-out; any slice after that is a retry.
        retry_targets = dispatched[3:]
        self.assertTrue(retry_targets, "expected at least one retry dispatch")
        self.assertNotIn("worker-a", retry_targets)

    def test_gemv_iteration_count_preserved_on_retry_slices(self) -> None:
        """``--iteration N`` value must flow through to redispatched slices."""
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        assignments = [
            _gemv_slice(worker_a, task_id="req-iter", row_start=0, row_end=50),
            _gemv_slice(worker_b, task_id="req-iter", row_start=50, row_end=100),
        ]

        simulator = _WorkerSimulator()
        event_a = simulator.configure(
            "worker-a", exception=ConnectionError("a died"),
        )
        event_b = simulator.configure("worker-b")
        event_a.set()
        event_b.set()

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = simulator.run
        handler = _make_handler(task_exchange)
        request = mock.Mock(
            request_id="req-iter", method=METHOD_GEMV, iteration_count=10,
        )

        outcomes = handler._run_assignments_with_failover(request, assignments)

        # worker-b's original slice + one retry slice for worker-a's 50 rows.
        self.assertEqual(len(outcomes), 2)
        for result, _ in outcomes:
            self.assertEqual(result.iteration_count, 10)
        # Every fake call, including the retry, received iteration_count=10.
        self.assertTrue(simulator.captured_iteration_counts)
        self.assertTrue(
            all(n == 10 for n in simulator.captured_iteration_counts),
            simulator.captured_iteration_counts,
        )

    def test_cascading_failure_retry_target_also_dies(self) -> None:
        """Retry slice lands on a worker that then dies too — still succeeds.

        Worker-a fails first; scheduler redispatches onto b and c. Worker-b
        then dies while processing its retry slice. The surviving worker c
        must absorb everything that was originally worker-a's range.
        """
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        worker_c = _make_worker_connection("worker-c")
        assignments = [
            _gemv_slice(worker_a, task_id="req-1", row_start=0, row_end=60),
            _gemv_slice(worker_b, task_id="req-1", row_start=60, row_end=120),
            _gemv_slice(worker_c, task_id="req-1", row_start=120, row_end=200),
        ]

        state_lock = threading.Lock()
        b_call_count = {"n": 0}

        def run_slice(request, assignment):
            runtime_id = assignment.connection.runtime_id
            if runtime_id == "worker-a":
                raise ConnectionError("worker-a down")
            if runtime_id == "worker-b":
                with state_lock:
                    b_call_count["n"] += 1
                    current = b_call_count["n"]
                # First call = original slice (success). Later calls = retry
                # landing on worker-b after worker-a died — this one crashes,
                # forcing a second redispatch onto worker-c.
                if current >= 2:
                    raise ConnectionError("worker-b also died during retry")
                return _gemv_result(assignment, request.iteration_count), _timing(assignment)
            return _gemv_result(assignment, request.iteration_count), _timing(assignment)

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = run_slice
        handler = _make_handler(task_exchange)
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV, iteration_count=1)

        outcomes = handler._run_assignments_with_failover(request, assignments)

        covered = sorted((result.row_start, result.row_end) for result, _ in outcomes)
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], 200)
        for left, right in zip(covered, covered[1:]):
            self.assertEqual(left[1], right[0], f"gap between {left} and {right}")
        dispatched = [
            call.args[1].connection.runtime_id
            for call in task_exchange.run_worker_task_slice.call_args_list
        ]
        # Once cascade resolves, only worker-c can be a retry target (a and b dead).
        self.assertGreaterEqual(dispatched.count("worker-c"), 2)

    def test_conv2d_mid_task_failure_produces_contiguous_channel_coverage(self) -> None:
        """Conv2d failover must tile the failed worker's output channels."""
        worker_a = _make_worker_connection("worker-a")
        worker_b = _make_worker_connection("worker-b")
        worker_c = _make_worker_connection("worker-c")
        assignments = [
            _conv2d_slice(worker_a, task_id="req-1", start_oc=0, end_oc=32),
            _conv2d_slice(worker_b, task_id="req-1", start_oc=32, end_oc=64),
            _conv2d_slice(worker_c, task_id="req-1", start_oc=64, end_oc=96),
        ]

        simulator = _WorkerSimulator()
        event_a = simulator.configure(
            "worker-a", exception=ConnectionError("conv2d worker-a down"),
        )
        event_b = simulator.configure("worker-b")
        event_c = simulator.configure("worker-c")
        event_a.set()
        event_b.set()
        event_c.set()

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = simulator.run
        handler = _make_handler(task_exchange)
        request = mock.Mock(request_id="req-1", method=METHOD_CONV2D, iteration_count=1)

        outcomes = handler._run_assignments_with_failover(request, assignments)

        channel_ranges = sorted(
            (result.start_oc, result.end_oc) for result, _ in outcomes
        )
        self.assertEqual(channel_ranges[0][0], 0)
        self.assertEqual(channel_ranges[-1][1], 96)
        for left, right in zip(channel_ranges, channel_ranges[1:]):
            self.assertEqual(left[1], right[0])

    def test_four_worker_shell_scenario_survives_one_crash(self) -> None:
        """Four concurrent 'shells' — kill one, verify the other three finish.

        This is the direct analog of the manual test: four worker processes
        running a gemv request with many iterations, one shell killed mid-run,
        the client still receives a valid aggregated answer.
        """
        workers = [_make_worker_connection(f"worker-{idx}") for idx in range(4)]
        assignments = [
            _gemv_slice(workers[0], task_id="req-1", row_start=0, row_end=25),
            _gemv_slice(workers[1], task_id="req-1", row_start=25, row_end=50),
            _gemv_slice(workers[2], task_id="req-1", row_start=50, row_end=75),
            _gemv_slice(workers[3], task_id="req-1", row_start=75, row_end=100),
        ]

        simulator = _WorkerSimulator()
        for worker in workers:
            simulator.configure(worker.runtime_id)
        # Victim: worker-2 (index 2) goes down mid-iteration.
        simulator._exceptions["worker-2"] = ConnectionError("shell 3 killed by operator")

        task_exchange = MagicMock()
        task_exchange.run_worker_task_slice.side_effect = simulator.run
        handler = _make_handler(task_exchange)
        request = mock.Mock(request_id="req-1", method=METHOD_GEMV, iteration_count=8)

        result_holder: dict = {}

        def _invoke() -> None:
            result_holder["outcomes"] = handler._run_assignments_with_failover(
                request, assignments,
            )

        runner = threading.Thread(target=_invoke)
        runner.start()

        # Ensure all four originals have entered run() before releasing them.
        deadline = time.monotonic() + 2.0
        while len(simulator.call_order) < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(
            set(simulator.call_order),
            {"worker-0", "worker-1", "worker-2", "worker-3"},
        )

        for runtime_id in ("worker-0", "worker-1", "worker-2", "worker-3"):
            simulator._proceed_events[runtime_id].set()

        runner.join(timeout=5.0)
        self.assertFalse(runner.is_alive())
        outcomes = result_holder["outcomes"]

        covered = sorted((result.row_start, result.row_end) for result, _ in outcomes)
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], 100)
        for left, right in zip(covered, covered[1:]):
            self.assertEqual(left[1], right[0])
        # Victim's original range [50, 75) must be covered by retry slices that
        # were never sent to worker-2.
        dispatched = [
            call.args[1].connection.runtime_id
            for call in task_exchange.run_worker_task_slice.call_args_list
        ]
        retry_targets = dispatched[4:]
        self.assertTrue(retry_targets)
        self.assertNotIn("worker-2", retry_targets)


if __name__ == "__main__":
    unittest.main()
