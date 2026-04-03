"""Platform detection tests."""

import unittest

from adapters.network import get_local_mac_address
from adapters.platform import detect_os, is_wsl


class PlatformTests(unittest.TestCase):
    """Validate platform reporting."""

    def test_detect_os_returns_supported_label(self) -> None:
        info = detect_os()
        self.assertIn(info.platform_name, {"windows", "linux", "macos", "wsl", "unknown"})
        self.assertIsInstance(info.is_admin, bool)

    def test_detect_os_matches_is_wsl(self) -> None:
        info = detect_os()
        self.assertEqual(info.is_wsl, is_wsl())

    def test_get_local_mac_address_is_printable(self) -> None:
        mac = get_local_mac_address()
        self.assertEqual(len(mac.split(":")), 6)


if __name__ == "__main__":
    unittest.main()
