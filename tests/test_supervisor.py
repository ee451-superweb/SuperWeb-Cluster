"""Supervisor lifecycle tests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.types import DiscoveryResult, FirewallStatus, PlatformInfo
from core.config import AppConfig
from supervision.supervisor import Supervisor


def _write_result_json(path: Path, *, usable_backends_by_method: dict[str, list[str]]) -> None:
    """Create a minimal result.json with the requested per-method backend lists."""
    payload = {
        "schema_version": 6,
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

    @mock.patch("supervision.supervisor.Supervisor.register_signal_handlers")
    @mock.patch("supervision.supervisor.Supervisor._join_main_node")
    @mock.patch("supervision.supervisor.Supervisor._discover_with_retries")
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

    @mock.patch("supervision.supervisor.Supervisor.register_signal_handlers")
    @mock.patch("supervision.supervisor.Supervisor._promote_to_main_node")
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

    @mock.patch("supervision.supervisor.time.sleep")
    @mock.patch("supervision.supervisor.random.uniform", return_value=1.75)
    @mock.patch("supervision.supervisor.run_pairing")
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

    def test_plan_capacity_compute_default_pins_gpu_and_spawns_cpu_peer(self) -> None:
        # A compute-only supervisor (not co-located with main-node) on a
        # GPU+CPU host exposes both backends: the GPU peer plus a CPU peer.
        # In multi-machine deployments supervisors don't know about each
        # other, so each host independently advertises its full capacity.
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
        self.assertEqual(supervisor._plan_capacity("compute"), ("cuda", "cpu"))

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
        fake_process.wait.return_value = 0
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_process) as popen_mock:
            supervisor._spawn_peer("cpu")
        if supervisor._peer_watcher_thread is not None:
            supervisor._peer_watcher_thread.join(timeout=2.0)
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
        fake_process = mock.Mock(pid=1)
        fake_process.wait.return_value = 0
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_process) as popen_mock, \
             mock.patch("supervision.supervisor.sys.platform", "win32"):
            supervisor._spawn_peer("cuda")
        if supervisor._peer_watcher_thread is not None:
            supervisor._peer_watcher_thread.join(timeout=2.0)
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
        with mock.patch("supervision.supervisor.cleanup_rules") as cleanup_mock:
            status = supervisor.shutdown()
        cleanup_mock.assert_not_called()
        self.assertFalse(status.applied)
        self.assertIn("delegated", status.message)

    def test_shutdown_calls_firewall_cleanup_when_not_peer_process(self) -> None:
        supervisor = self._build_supervisor(
            config=AppConfig(peer_process=False),
        )
        with mock.patch("supervision.supervisor.cleanup_rules") as cleanup_mock:
            cleanup_mock.return_value = FirewallStatus(
                supported=True,
                applied=True,
                needs_admin=False,
                backend="windows",
                message="cleaned",
            )
            supervisor.shutdown()
        cleanup_mock.assert_called_once()

    def test_spawn_peer_starts_watcher_thread_logging_unexpected_death(self) -> None:
        # When a peer dies on its own (no shutdown requested), the watcher
        # thread must emit a WARNING with the classified cause so operators
        # can tell an OS-normal eviction from a runner bug. No respawn.
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

        fake_proc = mock.Mock()
        fake_proc.pid = 4242
        fake_proc.wait.return_value = -11  # SIGSEGV equivalent on posix; classifier handles it
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_proc):
            supervisor._spawn_peer("cuda")
        watcher = supervisor._peer_watcher_thread
        self.assertIsNotNone(watcher)
        watcher.join(timeout=2.0)
        self.assertFalse(watcher.is_alive())
        warning_calls = supervisor.logger.warning.call_args_list
        self.assertTrue(
            any("died unexpectedly" in str(call) for call in warning_calls),
            f"expected unexpected-death WARNING, got {warning_calls!r}",
        )

    def test_spawn_peer_watcher_logs_info_when_terminate_requested(self) -> None:
        # If shutdown() called terminate first, the peer's exit is expected and
        # the watcher must log INFO, not WARNING — otherwise normal shutdowns
        # look like crashes in the audit log.
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

        wait_event = __import__("threading").Event()

        def _blocking_wait():
            wait_event.wait(timeout=2.0)
            return 0

        fake_proc = mock.Mock()
        fake_proc.pid = 4243
        fake_proc.wait.side_effect = _blocking_wait
        fake_proc.poll.return_value = None
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_proc):
            supervisor._spawn_peer("cuda")
        # Mark terminate requested before unblocking wait, mirroring shutdown().
        supervisor._peer_terminate_requested = True
        wait_event.set()
        supervisor._peer_watcher_thread.join(timeout=2.0)
        warning_calls = supervisor.logger.warning.call_args_list
        self.assertFalse(
            any("died unexpectedly" in str(call) for call in warning_calls),
            f"watcher must not warn on requested termination; got {warning_calls!r}",
        )

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
        fake_process = mock.Mock(pid=1)
        fake_process.wait.return_value = 0
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_process) as popen_mock, \
             mock.patch("supervision.supervisor.sys.platform", "win32"):
            supervisor._spawn_peer("cuda")
        if supervisor._peer_watcher_thread is not None:
            supervisor._peer_watcher_thread.join(timeout=2.0)
        kwargs = popen_mock.call_args.kwargs
        self.assertNotIn("stdout", kwargs)
        self.assertNotIn("stderr", kwargs)
        self.assertEqual(kwargs.get("creationflags"), 0x00000010)  # CREATE_NEW_CONSOLE

    def _spawned_supervisor(self, *, node_name: str = "main-node") -> Supervisor:
        """Build a supervisor with a fake spawned peer wired up.

        Why this helper: every eviction test needs the same shape — a supervisor
        that has called ``_spawn_peer`` so ``_peer_node_name`` and
        ``_peer_process`` are populated. Inlining this in each test obscured the
        thing the test actually exercises.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            _write_result_json(
                result_path,
                usable_backends_by_method={"gemv": ["cpu", "cuda"]},
            )
            supervisor = self._build_supervisor(
                config=AppConfig(node_name=node_name),
                result_path=result_path,
                bootstrap_script_path=Path("/tmp/bootstrap.py"),
            )
        return supervisor

class SupervisorPeerHeartbeatTests(SupervisorCapacityPlanningTests):
    """Validate the loopback heartbeat hang detector.

    The heartbeat is the supervisor's local hang detector: it works for any
    supervisor (including a second one that does not own a main-node
    registry) and fires on the first missed interval without waiting for any
    cluster-level signal. The tests here cover the supervisor side only —
    the listener and peer writer contracts are covered by
    ``test_supervisor_heartbeat.py``.
    """

    def _fake_listener(
        self, *, accept_result: bool, heartbeats: list[str]
    ) -> mock.Mock:
        """Return a ``SupervisorHeartbeatListener`` mock with scripted reads.

        Each entry in ``heartbeats`` is one of the tri-state constants
        (``HEARTBEAT_OK`` / ``HEARTBEAT_TIMEOUT`` / ``HEARTBEAT_CLOSED``).
        Once exhausted, further calls return ``HEARTBEAT_CLOSED`` so the
        watcher exits cleanly rather than spinning forever in tests.
        """
        from supervision.supervisor_heartbeat import HEARTBEAT_CLOSED

        listener = mock.Mock()
        listener.accept.return_value = accept_result
        remaining = list(heartbeats)

        def _wait(_timeout: float) -> str:
            return remaining.pop(0) if remaining else HEARTBEAT_CLOSED

        listener.wait_for_heartbeat.side_effect = _wait
        listener.port = 0
        return listener

    def _spawn_with_fake_listener(
        self, listener: mock.Mock
    ) -> tuple[Supervisor, mock.Mock]:
        """Build a supervisor that already has a peer_process and fake listener.

        Skips the real ``_spawn_peer`` because the heartbeat watcher's
        behavior is what we're testing; wiring up env vars and Popen kwargs
        is covered by the dedicated spawn test.
        """
        supervisor = self._spawned_supervisor()
        fake_process = mock.Mock(pid=4242)
        fake_process.wait.return_value = 0
        fake_process.poll.return_value = None  # alive
        supervisor._peer_process = fake_process
        supervisor._peer_backend = "cuda"
        supervisor._peer_node_name = "main-node-peer-cuda"
        supervisor._peer_heartbeat_listener = listener
        return supervisor, fake_process

    def test_spawn_peer_sets_env_var_and_starts_watcher(self) -> None:
        # Operators and the peer side both depend on the env var being set;
        # if it is missing, the peer silently runs without heartbeat and we
        # lose the hang detector. Pin both the var name and the port linkage.
        supervisor = self._spawned_supervisor()
        fake_process = mock.Mock(pid=9999)
        fake_process.wait.return_value = 0
        with mock.patch("supervision.supervisor.subprocess.Popen", return_value=fake_process) as popen_mock:
            supervisor._spawn_peer("cuda")
        # Keep the watcher from dangling on accept for the full 30s timeout.
        if supervisor._peer_heartbeat_listener is not None:
            supervisor._peer_heartbeat_listener.close()
        if supervisor._peer_heartbeat_thread is not None:
            supervisor._peer_heartbeat_thread.join(timeout=2.0)
        if supervisor._peer_watcher_thread is not None:
            supervisor._peer_watcher_thread.join(timeout=2.0)
        kwargs = popen_mock.call_args.kwargs
        self.assertIn("env", kwargs)
        self.assertIn("SUPERWEB_PEER_HEARTBEAT_PORT", kwargs["env"])
        port_str = kwargs["env"]["SUPERWEB_PEER_HEARTBEAT_PORT"]
        self.assertEqual(int(port_str), int(port_str))  # parses as int
        self.assertGreater(int(port_str), 0)

    def test_watcher_exits_silently_when_peer_never_connects(self) -> None:
        # Legacy peer builds and peers that crash before bootstrap will never
        # connect. The watcher must log once and get out of the way — it
        # must NOT kill the peer just because the heartbeat never established.
        listener = self._fake_listener(accept_result=False, heartbeats=[])
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        supervisor._watch_peer_heartbeat(listener, fake_process, "cuda")
        fake_process.terminate.assert_not_called()
        listener.wait_for_heartbeat.assert_not_called()

    def test_watcher_declares_hang_after_miss_threshold(self) -> None:
        # The load-bearing test: four consecutive TIMEOUT misses while
        # poll()==None is the definition of "hung", and the supervisor must
        # dump+kill. Note: TIMEOUT (not CLOSED) — the hang signal is the
        # silence of a peer whose socket is still open.
        from supervision.supervisor_heartbeat import HEARTBEAT_OK, HEARTBEAT_TIMEOUT

        listener = self._fake_listener(
            accept_result=True,
            heartbeats=[
                HEARTBEAT_OK,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_TIMEOUT,
            ],
        )
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        with mock.patch(
            "supervision.supervisor.dump_python_stack",
            return_value="Thread 0x1: blocked in cuda_dispatch",
        ) as dump_mock:
            supervisor._watch_peer_heartbeat(listener, fake_process, "cuda")
        dump_mock.assert_called_once_with(4242)
        fake_process.terminate.assert_called_once()

    def test_watcher_exits_silently_on_peer_socket_close(self) -> None:
        # Regression for the 2026-04-21 CUDA-runner incident: native runner
        # returned exit 1, peer exited, socket closed. The watcher used to
        # read four EOFs in the same millisecond and declare a hang. Must
        # now treat CLOSED as "peer is going away" and hand off to the
        # exit-code watcher without dumping or terminating.
        from supervision.supervisor_heartbeat import HEARTBEAT_CLOSED, HEARTBEAT_OK

        listener = self._fake_listener(
            accept_result=True,
            heartbeats=[HEARTBEAT_OK, HEARTBEAT_CLOSED],
        )
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        with mock.patch("supervision.supervisor.dump_python_stack") as dump_mock:
            supervisor._watch_peer_heartbeat(listener, fake_process, "cuda")
        dump_mock.assert_not_called()
        fake_process.terminate.assert_not_called()

    def test_watcher_resets_miss_count_on_resumed_heartbeat(self) -> None:
        # A transient GC pause or CPU spike can miss one or two heartbeats;
        # a recovery byte must clear the counter so transient stalls do not
        # kill otherwise-healthy peers.
        from supervision.supervisor_heartbeat import HEARTBEAT_OK, HEARTBEAT_TIMEOUT

        listener = mock.Mock()
        listener.accept.return_value = True
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        listener.wait_for_heartbeat.side_effect = _make_draining_side_effect(
            [
                HEARTBEAT_OK,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_OK,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_TIMEOUT,
                HEARTBEAT_OK,
            ],
            supervisor,
        )
        with mock.patch("supervision.supervisor.dump_python_stack") as dump_mock:
            supervisor._watch_peer_heartbeat(listener, fake_process, "cuda")
        dump_mock.assert_not_called()
        fake_process.terminate.assert_not_called()

    def test_watcher_does_not_fire_when_peer_exits_mid_sequence(self) -> None:
        # Covers the case where recv times out (socket still open) but the
        # process has in the meantime exited. poll() returning non-None
        # means "the exit-code watcher is about to / already has logged the
        # cause" — no dump, no terminate.
        from supervision.supervisor_heartbeat import HEARTBEAT_OK, HEARTBEAT_TIMEOUT

        listener = self._fake_listener(
            accept_result=True,
            heartbeats=[HEARTBEAT_OK, HEARTBEAT_TIMEOUT],
        )
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        fake_process.poll.side_effect = [None, 0]
        with mock.patch("supervision.supervisor.dump_python_stack") as dump_mock:
            supervisor._watch_peer_heartbeat(listener, fake_process, "cuda")
        dump_mock.assert_not_called()
        fake_process.terminate.assert_not_called()

    def test_terminate_peer_closes_heartbeat_listener(self) -> None:
        # Leaking a listening socket on shutdown would pin a loopback port
        # across restart and make the next spawn fail to bind on that port.
        listener = self._fake_listener(accept_result=True, heartbeats=[])
        supervisor, fake_process = self._spawn_with_fake_listener(listener)
        supervisor._terminate_peer()
        listener.close.assert_called_once()


def _make_draining_side_effect(script: list[str], supervisor: Supervisor):
    """Build a ``wait_for_heartbeat`` side effect that exits the watcher after the script drains.

    Why this helper: the watcher loop is infinite by design (it would run
    until the supervisor shuts down in production). In a unit test we need
    it to exit deterministically once the scripted sequence is consumed;
    flipping ``_shutdown_requested`` after the last scripted call is the
    smallest signal that matches the real exit path. ``script`` contains
    tri-state values from :mod:`supervision.supervisor_heartbeat`.
    """
    from supervision.supervisor_heartbeat import HEARTBEAT_CLOSED

    remaining = list(script)

    def _wait(_timeout: float) -> str:
        if not remaining:
            supervisor._shutdown_requested = True
            return HEARTBEAT_CLOSED
        value = remaining.pop(0)
        if not remaining:
            supervisor._shutdown_requested = True
        return value

    return _wait


if __name__ == "__main__":
    unittest.main()

