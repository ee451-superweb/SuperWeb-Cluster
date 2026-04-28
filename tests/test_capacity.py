"""Tests for ``supervision.capacity.load_usable_backends``."""

import json
import tempfile
import unittest
from pathlib import Path

from supervision.capacity import load_usable_backends


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class LoadUsableBackendsTests(unittest.TestCase):
    """Validate result.json parsing for backend availability discovery."""

    def test_unions_usable_backends_across_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write(
                result_path,
                {
                    "methods": {
                        "gemv": {"usable_backends": ["cpu", "cuda"]},
                        "conv2d": {"usable_backends": ["cuda", "metal"]},
                    },
                },
            )
            self.assertEqual(
                load_usable_backends(result_path),
                {"cpu", "cuda", "metal"},
            )

    def test_missing_file_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            absent = Path(temp_dir) / "absent.json"
            self.assertEqual(load_usable_backends(absent), set())

    def test_malformed_json_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(load_usable_backends(result_path), set())

    def test_missing_methods_section_returns_empty_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write(result_path, {"schema_version": 6})
            self.assertEqual(load_usable_backends(result_path), set())

    def test_non_string_backend_entries_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write(
                result_path,
                {
                    "methods": {
                        "gemv": {"usable_backends": ["cpu", None, 42, "cuda"]},
                    },
                },
            )
            self.assertEqual(load_usable_backends(result_path), {"cpu", "cuda"})


if __name__ == "__main__":
    unittest.main()
