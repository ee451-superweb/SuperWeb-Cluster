"""Unit tests for the native runner death-cause log helpers.

These helpers exist so a CalledProcessError from the conv2d / gemv native
runners produces an actionable log line (exit code + classification + stderr
tail) before the exception propagates up through the worker pool. Without
them the worker would die silently because its CREATE_NEW_CONSOLE stderr
disappears with the console window.
"""

import logging
import unittest
from unittest import mock


class Conv2dRunnerFailureLogTests(unittest.TestCase):
    """Exercise the conv2d executor's native-runner failure helpers."""

    def test_log_runner_failure_includes_classification_and_stderr_tail(self) -> None:
        from compute_node.compute_methods.conv2d import executor

        task = mock.Mock()
        task.task_id = "conv2d-task-7"
        with mock.patch.object(executor, "_LOGGER") as logger:
            executor._log_runner_failure(
                method="conv2d",
                backend_name="cuda",
                task=task,
                returncode=-1073741819,  # 0xC0000005 sign-extended
                stderr="cudaMalloc failed: out of memory",
                stdout="",
                elapsed_ms=2500,
            )
        self.assertEqual(logger.error.call_count, 1)
        rendered = logger.error.call_args[0][0] % logger.error.call_args[0][1:]
        self.assertIn("backend=cuda", rendered)
        self.assertIn("task_id=conv2d-task-7", rendered)
        self.assertIn("STATUS_ACCESS_VIOLATION", rendered)
        self.assertIn("cudaMalloc failed", rendered)

    def test_log_runner_timeout_emits_error(self) -> None:
        from compute_node.compute_methods.conv2d import executor

        task = mock.Mock()
        task.task_id = "conv2d-task-9"
        with mock.patch.object(executor, "_LOGGER") as logger:
            executor._log_runner_timeout(
                method="conv2d",
                backend_name="cpu",
                task=task,
                timeout=900.0,
                stderr=b"hang",
                stdout=None,
            )
        self.assertEqual(logger.error.call_count, 1)
        rendered = logger.error.call_args[0][0] % logger.error.call_args[0][1:]
        self.assertIn("timed out", rendered)
        self.assertIn("timeout=900", rendered)
        self.assertIn("hang", rendered)

    def test_tail_stream_truncates_long_payload(self) -> None:
        from compute_node.compute_methods.conv2d import executor

        long_text = "x" * 10000
        tail = executor._tail_stream(long_text, limit=100)
        self.assertTrue(tail.endswith("x" * 100))
        self.assertIn("truncated", tail)

    def test_tail_stream_handles_none(self) -> None:
        from compute_node.compute_methods.conv2d import executor

        self.assertEqual(executor._tail_stream(None), "<none>")
        self.assertEqual(executor._tail_stream(""), "<empty>")
        self.assertEqual(executor._tail_stream(b""), "<empty>")

    def test_runner_process_error_carries_stderr_through_pickle(self) -> None:
        # ProcessPoolExecutor pickles exceptions to send them back to the
        # parent. The 2026-04-21 incident showed CalledProcessError losing its
        # stderr in transit; RunnerProcessError must carry stderr inside its
        # str() so the parent's TASK_FAIL builder and worker logger surface it.
        import pickle

        from compute_node.compute_methods.conv2d import executor

        task = mock.Mock()
        task.task_id = "conv2d-task-2"
        message = executor._format_runner_failure_message(
            method="conv2d",
            backend_name="cuda",
            task=task,
            returncode=1,
            stderr="cudaMalloc failed: out of memory",
            stdout="",
            elapsed_ms=2000,
        )
        original = executor.RunnerProcessError(message)
        revived = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(revived, executor.RunnerProcessError)
        self.assertIn("cudaMalloc failed", str(revived))
        self.assertIn("task_id=conv2d-task-2", str(revived))
        self.assertIn("returncode=1", str(revived))


class GemvRunnerFailureLogTests(unittest.TestCase):
    """Exercise the gemv task executor's native-runner failure helpers."""

    def test_log_gemv_runner_failure_includes_row_range(self) -> None:
        from compute_node import task_executor

        with mock.patch.object(task_executor, "_LOGGER") as logger:
            task_executor._log_gemv_runner_failure(
                backend_name="cpu",
                task_id="gemv-task-3",
                row_start=512,
                row_end=1024,
                returncode=1,
                stderr="error: bad alloc",
                stdout="some progress",
            )
        self.assertEqual(logger.error.call_count, 1)
        rendered = logger.error.call_args[0][0] % logger.error.call_args[0][1:]
        self.assertIn("backend=cpu", rendered)
        self.assertIn("task_id=gemv-task-3", rendered)
        self.assertIn("rows=[512, 1024)", rendered)
        self.assertIn("bad alloc", rendered)


if __name__ == "__main__":
    unittest.main()
