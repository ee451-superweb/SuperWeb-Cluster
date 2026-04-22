"""Supervisor lifecycle tests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from common.types import DiscoveryResult, FirewallStatus, PlatformInfo
from app.config import AppConfig
from app.supervisor import Supervisor


def _write_result_json(path: Path, *, usable_backends_by_method: dict[str, list[str]]) -> None:
    """Create a minimal result.json with the requested per-method backend lists."""
    payload = {
        "schema_version": 5,
        "methods": {
            method_name: {"usable_backends": list(backends)}
            for method_name, backends in usable_backends_by_method.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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
                hostname="mdong-zephy14",
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
                hostname="mdong-zephy14",
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


class SupervisorCapacityPlanningTests(unittest.TestCase):
    """Validate flag validation and peer-spawn planning against result.json."""

    def _build_supervisor(
        self,
        *,
        config: AppConfig,
        result_path: Path | None = None,
        bootstrap_script_path: Path | None = None,
    ) -> Supervisor:
        return Supervisor(
            config=config,
            platform_info=PlatformInfo(
                platform_name="windows",
                system="Windows",
                release="11",
                machine="AMD64",
                hostname="mdong-zephy14",
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
            benchmark_result_path=result_path,
            bootstrap_script_path=bootstrap_script_path,
        )

    def test_init_rejects_pinned_backend_absent_from_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(result_path, usable_backends_by_method={"gemv": ["cpu"]})
            with self.assertRaises(RuntimeError) as ctx:
                self._build_supervisor(
                    config=AppConfig(pinned_backend="cuda"),
                    result_path=result_path,
                )
        self.assertIn("cuda", str(ctx.exception))
        self.assertIn("cpu", str(ctx.exception))

    def test_init_rejects_pinned_backend_when_result_json_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "absent.json"
            with self.assertRaises(RuntimeError) as ctx:
                self._build_supervisor(
                    config=AppConfig(pinned_backend="cpu"),
                    result_path=missing,
                )
        self.assertIn("re-run the local benchmark", str(ctx.exception))

    def test_init_accepts_pinned_backend_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"], "conv2d": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(pinned_backend="cuda"),
                result_path=result_path,
            )
        self.assertEqual(supervisor.available_backends, frozenset({"cpu", "cuda"}))

    def test_plan_capacity_main_with_dual_purpose_picks_best_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(dual_purpose=True),
                result_path=result_path,
            )
        self.assertEqual(supervisor._plan_capacity("main"), (None, "cuda"))

    def test_plan_capacity_main_with_dual_purpose_warns_when_no_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(result_path, usable_backends_by_method={"gemv": ["cpu"]})
            supervisor = self._build_supervisor(
                config=AppConfig(dual_purpose=True),
                result_path=result_path,
            )
            self.assertEqual(supervisor._plan_capacity("main"), (None, None))
        supervisor.logger.warning.assert_called_once()

    def test_plan_capacity_main_without_dual_purpose_no_peer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(dual_purpose=False),
                result_path=result_path,
            )
        self.assertEqual(supervisor._plan_capacity("main"), (None, None))

    def test_plan_capacity_compute_default_pins_gpu_and_does_not_spawn_cpu_peer(self) -> None:
        # When a compute-node host has both GPU and CPU available, the local
        # CPU is held by GPU driver overhead and must NOT be exposed as a
        # second backend. Result-ranking should pick a different host's CPU.
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
            )
        self.assertEqual(supervisor._plan_capacity("compute"), ("cuda", None))

    def test_plan_capacity_compute_default_no_gpu_pins_cpu_no_peer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(result_path, usable_backends_by_method={"gemv": ["cpu"]})
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
            )
        self.assertEqual(supervisor._plan_capacity("compute"), ("cpu", None))

    def test_plan_capacity_compute_explicit_pin_no_peer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(pinned_backend="cpu"),
                result_path=result_path,
            )
        self.assertEqual(supervisor._plan_capacity("compute"), ("cpu", None))

    def test_peer_command_forwards_network_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(
                    node_name="main-node",
                    multicast_group="239.1.2.3",
                    udp_port=5454,
                    tcp_port=52100,
                    data_plane_port=52101,
                ),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        command = supervisor._peer_command("cuda")
        self.assertIn("--role", command)
        self.assertEqual(command[command.index("--role") + 1], "discover")
        self.assertEqual(command[command.index("--backend") + 1], "cuda")
        self.assertEqual(command[command.index("--node-name") + 1], "main-node-peer-cuda")
        self.assertEqual(command[command.index("--multicast-group") + 1], "239.1.2.3")
        self.assertEqual(command[command.index("--udp-port") + 1], "5454")
        self.assertEqual(command[command.index("--tcp-port") + 1], "52100")
        self.assertEqual(command[command.index("--data-plane-port") + 1], "52101")
        self.assertIn("--no-manual-fallback", command)

    def test_spawn_peer_records_process_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        fake_process = mock.Mock(pid=4242)
        with mock.patch("app.supervisor.subprocess.Popen", return_value=fake_process) as popen_mock:
            supervisor._spawn_peer("cpu")
        popen_mock.assert_called_once()
        self.assertIs(supervisor._peer_process, fake_process)
        self.assertEqual(supervisor._peer_backend, "cpu")

    def test_terminate_peer_sends_terminate_then_waits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        fake_process = mock.Mock(pid=4242)
        fake_process.poll.return_value = None
        fake_process.wait.return_value = 0
        supervisor._peer_process = fake_process
        supervisor._peer_backend = "cpu"

        supervisor._terminate_peer()

        fake_process.terminate.assert_called_once()
        fake_process.wait.assert_called_once()
        self.assertIsNone(supervisor._peer_process)

    def test_peer_command_appends_no_cli_flag_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(no_cli=True),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        command = supervisor._peer_command("cuda")
        self.assertIn("--no-cli", command)

    def test_peer_command_omits_no_cli_flag_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        command = supervisor._peer_command("cuda")
        self.assertNotIn("--no-cli", command)

    def test_spawn_peer_uses_devnull_and_detached_flags_when_no_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(no_cli=True),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        with mock.patch("app.supervisor.subprocess.Popen", return_value=mock.Mock(pid=1)) as popen_mock, \
             mock.patch("app.supervisor.sys.platform", "win32"):
            supervisor._spawn_peer("cuda")
        kwargs = popen_mock.call_args.kwargs
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertIn("creationflags", kwargs)
        self.assertTrue(kwargs["creationflags"] & 0x00000008)  # DETACHED_PROCESS

    def test_peer_command_appends_peer_process_flag(self) -> None:
        # Spawned peers must carry --peer-process so their bootstrap delegates
        # firewall lifecycle to the parent supervisor instead of touching the
        # global rule names that the parent owns.
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        command = supervisor._peer_command("cuda")
        self.assertIn("--peer-process", command)

    def test_shutdown_skips_firewall_cleanup_when_peer_process(self) -> None:
        # A peer-process supervisor must not call cleanup_rules() because the
        # firewall rule names are global, and removing them would tear down
        # the parent supervisor's still-needed rules.
        supervisor = self._build_supervisor(
            config=AppConfig(peer_process=True),
        )
        with mock.patch("app.supervisor.cleanup_rules") as cleanup_mock:
            status = supervisor.shutdown()
        cleanup_mock.assert_not_called()
        self.assertFalse(status.applied)
        self.assertIn("delegated", status.message)

    def test_shutdown_calls_firewall_cleanup_when_not_peer_process(self) -> None:
        supervisor = self._build_supervisor(
            config=AppConfig(peer_process=False),
        )
        with mock.patch("app.supervisor.cleanup_rules") as cleanup_mock:
            cleanup_mock.return_value = FirewallStatus(
                supported=True,
                applied=True,
                needs_admin=False,
                backend="windows",
                message="cleaned",
            )
            supervisor.shutdown()
        cleanup_mock.assert_called_once()

    def test_spawn_peer_uses_create_new_console_by_default_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        with mock.patch("app.supervisor.subprocess.Popen", return_value=mock.Mock(pid=1)) as popen_mock, \
             mock.patch("app.supervisor.sys.platform", "win32"):
            supervisor._spawn_peer("cuda")
        kwargs = popen_mock.call_args.kwargs
        self.assertNotIn("stdout", kwargs)
        self.assertNotIn("stderr", kwargs)
        self.assertEqual(kwargs.get("creationflags"), 0x00000010)  # CREATE_NEW_CONSOLE


if __name__ == "__main__":
    unittest.main()

