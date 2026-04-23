"""Validate that drain_completed_tasks survives any task-level exception.

Why this exists: a previous narrow ``except (OSError, RuntimeError, ValueError)``
allowed ``CalledProcessError`` (and others) raised by native runners to escape
the worker loop, killing the compute-node silently. The worker must instead
report TASK_FAIL upstream and keep draining.
"""

import subprocess
import unittest
from concurrent.futures import Future
from unittest import mock

from core.config import AppConfig
from compute_node.worker_services import WorkerTaskRuntimeService


class _StubSession:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def send(self, message) -> None:
        self.sent.append(message)


def _make_task(task_id: str = "t-001"):
    task = mock.Mock()
    task.task_id = task_id
    task.request_id = "r-001"
    task.method = "gemv"
    return task


def _completed_future_with_exception(exc: BaseException) -> Future:
    fut: Future = Future()
    fut.set_exception(exc)
    return fut


class DrainCompletedTasksExceptionTests(unittest.TestCase):
    """Ensure unexpected exceptions become TASK_FAIL instead of killing the worker."""

    def _service(self) -> WorkerTaskRuntimeService:
        return WorkerTaskRuntimeService(
            config=AppConfig(),
            logger=mock.Mock(),
            node_name="worker-test",
        )

    def _drain_with_exception(self, exc: BaseException) -> _StubSession:
        service = self._service()
        task = _make_task()
        future = _completed_future_with_exception(exc)
        pending = {task.task_id: (task, future)}
        session = _StubSession()
        service.drain_completed_tasks(
            session=session,
            assigned_node_id="node-1",
            pending_tasks=pending,
            artifact_manager=None,
        )
        # Worker must drop the task from pending and not re-raise.
        self.assertEqual(pending, {})
        return session

    def test_called_process_error_becomes_task_fail(self) -> None:
        # CalledProcessError used to escape the narrow catch and kill the
        # worker. Now it must turn into a TASK_FAIL with a useful error string.
        exc = subprocess.CalledProcessError(
            returncode=-1073741819,  # 0xC0000005 sign-extended (Windows AV)
            cmd=["runner.exe"],
            stderr="boom",
        )
        session = self._drain_with_exception(exc)
        self.assertEqual(len(session.sent), 1)
        self.assertIn("CalledProcessError", str(session.sent[0]))

    def test_arbitrary_exception_becomes_task_fail(self) -> None:
        # Any unexpected exception type must still be translated rather than
        # propagating; this is the regression guard for narrow excepts.
        session = self._drain_with_exception(ZeroDivisionError("pool died mid-task"))
        self.assertEqual(len(session.sent), 1)
        self.assertIn("ZeroDivisionError", str(session.sent[0]))

    def test_keyboard_interrupt_still_propagates(self) -> None:
        # KeyboardInterrupt is the operator pressing Ctrl+C; we want shutdown,
        # not TASK_FAIL silence. The catch broadens but explicitly re-raises.
        service = self._service()
        task = _make_task()
        future = _completed_future_with_exception(KeyboardInterrupt())
        pending = {task.task_id: (task, future)}
        session = _StubSession()
        with self.assertRaises(KeyboardInterrupt):
            service.drain_completed_tasks(
                session=session,
                assigned_node_id="node-1",
                pending_tasks=pending,
                artifact_manager=None,
            )


if __name__ == "__main__":
    unittest.main()
