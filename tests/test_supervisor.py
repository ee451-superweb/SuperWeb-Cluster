"""Supervisor lifecycle tests."""

import unittest
from unittest import mock

from common.types import DiscoveryResult, FirewallStatus, PlatformInfo
from app.config import AppConfig
from app.supervisor import Supervisor


class SupervisorTests(unittest.TestCase):
    """Validate discovery/runtime transitions."""

    def _build_supervisor(self, role: str = "discover") -> Supervisor:
        return Supervisor(
            config=AppConfig(role=role),
            platform_info=PlatformInfo(
                platform_name="windows",
                system="Windows",
                release="11",
                machine="AMD64",
                is_wsl=False,
                is_admin=False,
                can_elevate=True,
            ),
            firewall_status=FirewallStatus(
                supported=True,
                applied=False,
                needs_admin=False,
                backend="windows",
                message="ok",
            ),
            logger=mock.Mock(),
        )

    @mock.patch("app.supervisor.Supervisor.register_signal_handlers")
    @mock.patch("app.supervisor.Supervisor._join_main_node")
    @mock.patch("app.supervisor.Supervisor._discover_with_retries")
    def test_discover_success_enters_compute_runtime(
        self,
        discover_with_retries_mock: mock.Mock,
        join_main_node_mock: mock.Mock,
        register_signal_handlers_mock: mock.Mock,
    ) -> None:
        del register_signal_handlers_mock
        discover_with_retries_mock.return_value = DiscoveryResult(
            success=True,
            peer_address="10.0.0.5",
            peer_port=52020,
            source="mdns",
            message="ok",
        )
        join_main_node_mock.return_value = DiscoveryResult(success=True, message="runtime ok")
        supervisor = self._build_supervisor(role="discover")

        result = supervisor.run()

        self.assertTrue(result.success)
        join_main_node_mock.assert_called_once()

    @mock.patch("app.supervisor.Supervisor.register_signal_handlers")
    @mock.patch("app.supervisor.Supervisor._promote_to_main_node")
    def test_announce_role_enters_main_node_runtime(
        self,
        promote_to_main_node_mock: mock.Mock,
        register_signal_handlers_mock: mock.Mock,
    ) -> None:
        del register_signal_handlers_mock
        promote_to_main_node_mock.return_value = DiscoveryResult(success=True, message="scheduler ok")
        supervisor = self._build_supervisor(role="announce")

        result = supervisor.run()

        self.assertTrue(result.success)
        promote_to_main_node_mock.assert_called_once()

    @mock.patch("app.supervisor.time.sleep")
    @mock.patch("app.supervisor.random.uniform", return_value=1.75)
    @mock.patch("app.supervisor.run_pairing")
    def test_discover_with_retries_uses_jittered_sleep_between_attempts(
        self,
        run_pairing_mock: mock.Mock,
        random_uniform_mock: mock.Mock,
        sleep_mock: mock.Mock,
    ) -> None:
        run_pairing_mock.side_effect = [
            DiscoveryResult(success=False, message="timeout"),
            DiscoveryResult(success=False, message="timeout"),
            DiscoveryResult(success=True, peer_address="10.0.0.5", peer_port=52020, source="mdns", message="ok"),
        ]
        supervisor = Supervisor(
            config=AppConfig(role="discover", discovery_attempts=3, discovery_retry_delay=2.0),
            platform_info=PlatformInfo(
                platform_name="windows",
                system="Windows",
                release="11",
                machine="AMD64",
                is_wsl=False,
                is_admin=False,
                can_elevate=True,
            ),
            firewall_status=FirewallStatus(
                supported=True,
                applied=False,
                needs_admin=False,
                backend="windows",
                message="ok",
            ),
            logger=mock.Mock(),
        )

        result = supervisor._discover_with_retries()

        self.assertTrue(result.success)
        self.assertEqual(run_pairing_mock.call_count, 3)
        self.assertEqual(random_uniform_mock.call_count, 2)
        random_uniform_mock.assert_any_call(1.0, 3.0)
        self.assertEqual(sleep_mock.call_args_list, [mock.call(1.75), mock.call(1.75)])


if __name__ == "__main__":
    unittest.main()

