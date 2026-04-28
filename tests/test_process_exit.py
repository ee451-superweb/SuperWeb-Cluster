"""Validate the process exit-code classifier used in observability logs."""

import signal
import unittest
from unittest import mock

from core.process_exit import classify_exit_code


class ClassifyExitCodeTests(unittest.TestCase):
    """Verify operator-readable explanations for child-process exit codes."""

    def test_none_returncode_reports_still_running(self) -> None:
        self.assertEqual(classify_exit_code(None), "still running")

    def test_zero_returncode_reports_clean_exit(self) -> None:
        self.assertEqual(classify_exit_code(0), "exit 0 (clean)")

    @mock.patch("core.process_exit.sys.platform", "win32")
    def test_windows_known_status_code_named(self) -> None:
        # 0xC0000005 is the most common cause-of-death on Windows: a segfault
        # from a native runner. Operators must be able to tell that apart from
        # an OS-normal Ctrl+C kill.
        message = classify_exit_code(0xC0000005)
        self.assertIn("STATUS_ACCESS_VIOLATION", message)
        self.assertIn("0xC0000005", message)

    @mock.patch("core.process_exit.sys.platform", "win32")
    def test_windows_signed_negative_status_normalized_to_unsigned(self) -> None:
        # Python sometimes surfaces NTSTATUS values as negative ints once they
        # round-trip through ctypes; the classifier must normalize so the table
        # lookup still hits.
        signed = -1073741819  # == 0xC0000005 sign-extended
        message = classify_exit_code(signed)
        self.assertIn("STATUS_ACCESS_VIOLATION", message)

    @mock.patch("core.process_exit.sys.platform", "win32")
    def test_windows_unknown_status_code_reports_fault(self) -> None:
        message = classify_exit_code(0xC0001234)
        self.assertIn("0xC0001234", message)
        self.assertIn("NTSTATUS-style fault", message)

    @mock.patch("core.process_exit.sys.platform", "win32")
    def test_windows_small_nonzero_reports_runner_failure(self) -> None:
        message = classify_exit_code(1)
        self.assertIn("exit 1", message)
        self.assertIn("runner-reported failure", message)

    @mock.patch("core.process_exit.sys.platform", "linux")
    def test_posix_negative_returncode_reports_signal(self) -> None:
        message = classify_exit_code(-signal.SIGSEGV.value)
        self.assertIn("SIGSEGV", message)
        self.assertIn(f"signal {signal.SIGSEGV.value}", message)

    @mock.patch("core.process_exit.sys.platform", "linux")
    def test_posix_sigkill_message_mentions_oom_hint(self) -> None:
        message = classify_exit_code(-9)
        self.assertIn("SIGKILL", message)
        self.assertIn("OOM", message)


if __name__ == "__main__":
    unittest.main()
