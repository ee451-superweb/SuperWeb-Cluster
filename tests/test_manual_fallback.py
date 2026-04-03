"""Manual fallback tests."""

import unittest
from unittest import mock

from constants import DEFAULT_TCP_PORT
from discovery.fallback import prompt_manual_address


class ManualFallbackTests(unittest.TestCase):
    """Validate manual host parsing."""

    def test_manual_input_returns_peer(self) -> None:
        with mock.patch("builtins.input", return_value="127.0.0.1:9999"):
            result = prompt_manual_address(default_port=DEFAULT_TCP_PORT)
        self.assertTrue(result.success)
        self.assertEqual(result.peer_address, "127.0.0.1")
        self.assertEqual(result.peer_port, 9999)


if __name__ == "__main__":
    unittest.main()
