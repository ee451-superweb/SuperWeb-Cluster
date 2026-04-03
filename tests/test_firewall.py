"""Firewall adapter dispatch tests."""

import unittest
from subprocess import CompletedProcess
from unittest import mock

from adapters.firewall import cleanup_rules, ensure_rules
from adapters.firewall import windows as windows_firewall
from common.types import PlatformInfo
from constants import DEFAULT_DISCOVERY_PORT, WINDOWS_FIREWALL_RULE_NAME


class FirewallTests(unittest.TestCase):
    """Validate firewall adapter dispatch behavior."""

    def test_unknown_platform_returns_unsupported_status(self) -> None:
        platform_info = PlatformInfo(
            platform_name="unknown",
            system="UnknownOS",
            release="0",
            machine="x86_64",
            is_wsl=False,
            is_admin=False,
            can_elevate=False,
        )
        status = ensure_rules(platform_info, DEFAULT_DISCOVERY_PORT)
        self.assertFalse(status.supported)
        cleanup_status = cleanup_rules(platform_info, DEFAULT_DISCOVERY_PORT)
        self.assertFalse(cleanup_status.supported)

    def test_windows_ensure_rules_requires_admin(self) -> None:
        status = windows_firewall.ensure_rules(discovery_port=DEFAULT_DISCOVERY_PORT, is_admin_user=False)
        self.assertTrue(status.needs_admin)
        self.assertFalse(status.applied)

    @mock.patch("adapters.firewall.windows.subprocess.run")
    def test_windows_ensure_rules_adds_inbound_and_outbound_rules(
        self,
        subprocess_run_mock: mock.Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="inbound ok", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="outbound ok", stderr=""),
        ]

        status = windows_firewall.ensure_rules(discovery_port=DEFAULT_DISCOVERY_PORT, is_admin_user=True)

        self.assertTrue(status.applied)
        self.assertEqual(subprocess_run_mock.call_count, 2)

        inbound_command = subprocess_run_mock.call_args_list[0].args[0]
        outbound_command = subprocess_run_mock.call_args_list[1].args[0]
        self.assertIn("dir=in", inbound_command)
        self.assertIn(f"localport={DEFAULT_DISCOVERY_PORT}", inbound_command)
        self.assertIn("profile=any", inbound_command)
        self.assertIn("dir=out", outbound_command)
        self.assertIn(f"remoteport={DEFAULT_DISCOVERY_PORT}", outbound_command)
        self.assertIn("profile=any", outbound_command)

    @mock.patch("adapters.firewall.windows.subprocess.run")
    def test_windows_cleanup_rules_deletes_both_rules(
        self,
        subprocess_run_mock: mock.Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="deleted inbound", stderr=""),
            CompletedProcess(args=[], returncode=0, stdout="deleted outbound", stderr=""),
        ]

        status = windows_firewall.cleanup_rules(discovery_port=DEFAULT_DISCOVERY_PORT, is_admin_user=True)

        self.assertTrue(status.applied)
        self.assertEqual(subprocess_run_mock.call_count, 2)
        inbound_command = subprocess_run_mock.call_args_list[0].args[0]
        outbound_command = subprocess_run_mock.call_args_list[1].args[0]
        self.assertIn("delete", inbound_command)
        self.assertIn("delete", outbound_command)
        self.assertIn(f"name={WINDOWS_FIREWALL_RULE_NAME}-Inbound", inbound_command)
        self.assertIn(f"name={WINDOWS_FIREWALL_RULE_NAME}-Outbound", outbound_command)


if __name__ == "__main__":
    unittest.main()
