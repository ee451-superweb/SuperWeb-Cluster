"""Logging setup tests."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import DEFAULT_LOG_FILE_MAX_BYTES
from core.logging_setup import archive_existing_logs, cleanse_existing_logs, configure_logging, rebind_logging_role


class LoggingSetupTests(unittest.TestCase):
    """Validate role-aware rotating audit log configuration."""

    def _close_root_handlers(self) -> None:
        """Close root logging handlers so temporary log files can be cleaned up."""

        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()

    def tearDown(self) -> None:
        self._close_root_handlers()

    def test_configure_logging_creates_role_prefixed_rotating_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with (
                mock.patch("core.logging_setup._SESSION_TIMESTAMP", "20260417-120000"),
                mock.patch("core.logging_setup._log_directory", return_value=log_dir),
            ):
                logger = configure_logging(role="main")
                logger.info("hello from main")

            root_logger = logging.getLogger()
            self.assertEqual(len(root_logger.handlers), 1)
            handler = root_logger.handlers[0]
            self.assertEqual(Path(handler.baseFilename).name, "main-20260417-120000.txt")
            self.assertEqual(handler.maxBytes, DEFAULT_LOG_FILE_MAX_BYTES)
            self.assertTrue((log_dir / "main-20260417-120000.txt").exists())
            self._close_root_handlers()

    def test_rebind_logging_role_switches_to_new_role_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with (
                mock.patch("core.logging_setup._SESSION_TIMESTAMP", "20260417-120100"),
                mock.patch("core.logging_setup._log_directory", return_value=log_dir),
            ):
                configure_logging(role="bootstrap")
                logger = rebind_logging_role("worker")
                logger.info("hello from worker")

            handler = logging.getLogger().handlers[0]
            self.assertEqual(Path(handler.baseFilename).name, "worker-20260417-120100.txt")
            self.assertTrue((log_dir / "worker-20260417-120100.txt").exists())
            self._close_root_handlers()

    def test_archive_existing_logs_zips_loose_logs_and_removes_originals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "bootstrap-20260417-110000.txt").write_text("bootstrap\n", encoding="utf-8")
            (log_dir / "worker-20260417-110001.txt.1").write_text("worker\n", encoding="utf-8")
            (log_dir / "logs-archive-older.zip").write_bytes(b"zip")
            (log_dir / "scratch.bin").write_bytes(b"binary")

            with (
                mock.patch("core.logging_setup._SESSION_TIMESTAMP", "20260417-120200"),
                mock.patch("core.logging_setup._log_directory", return_value=log_dir),
            ):
                archive_path, archived_count = archive_existing_logs()

            self.assertEqual(archived_count, 2)
            self.assertIsNotNone(archive_path)
            self.assertTrue(archive_path.exists())
            self.assertFalse((log_dir / "bootstrap-20260417-110000.txt").exists())
            self.assertFalse((log_dir / "worker-20260417-110001.txt.1").exists())
            self.assertTrue((log_dir / "logs-archive-older.zip").exists())
            self.assertTrue((log_dir / "scratch.bin").exists())
            with zipfile.ZipFile(archive_path, "r") as archive_file:
                self.assertEqual(
                    sorted(archive_file.namelist()),
                    ["bootstrap-20260417-110000.txt", "worker-20260417-110001.txt.1"],
                )
                expected_compression = getattr(zipfile, "ZIP_ZSTANDARD", zipfile.ZIP_DEFLATED)
                for zip_info in archive_file.infolist():
                    self.assertEqual(zip_info.compress_type, expected_compression)

    def test_cleanse_existing_logs_removes_logs_and_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "bootstrap-20260417-110000.txt").write_text("bootstrap\n", encoding="utf-8")
            (log_dir / "worker-20260417-110001.txt.1").write_text("worker\n", encoding="utf-8")
            (log_dir / "logs-archive-older.zip").write_bytes(b"zip")
            (log_dir / "scratch.bin").write_bytes(b"binary")

            with mock.patch("core.logging_setup._log_directory", return_value=log_dir):
                removed_count = cleanse_existing_logs()

            self.assertEqual(removed_count, 3)
            self.assertEqual([path.name for path in log_dir.iterdir()], ["scratch.bin"])


if __name__ == "__main__":
    unittest.main()
