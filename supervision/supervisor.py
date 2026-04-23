"""Minimal supervisor for superweb-cluster Sprint 1."""

from __future__ import annotations

import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

_PRE_DISCOVERY_STAGGER_MAX_SECONDS = 1.5
_PRE_DISCOVERY_STAGGER_SLOTS = 1024
_PEER_TERMINATE_TIMEOUT_SECONDS = 5.0
_PEER_HEARTBEAT_ACCEPT_TIMEOUT_SECONDS = 30.0

from adapters.firewall import cleanup_rules
from adapters.audit_log import write_audit_event
from adapters.process import python_utf8_command
from supervision.capacity import load_usable_backends
from core.constants import (
    BACKEND_CPU,
    GPU_BACKEND_PRIORITY,
)
from supervision.peer_diagnostics import dump_python_stack
from supervision.supervisor_heartbeat import (
    HEARTBEAT_CLOSED,
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_MISS_THRESHOLD,
    HEARTBEAT_OK,
    HEARTBEAT_PORT_ENV,
    SupervisorHeartbeatListener,
)
from core.process_exit import classify_exit_code
from core.state import RuntimeState
from core.types import DiscoveryResult, FirewallStatus, PlatformInfo
from compute_node.worker_loop import ComputeNodeRuntime
from core.config import AppConfig
from core.logging_setup import rebind_logging_role
from discovery.fallback import prompt_manual_address
from discovery.pairing import run_pairing
from main_node.control_loop import MainNodeRuntime
from core.tracing import trace_function


class Supervisor:
    """Coordinate the kickoff discovery flow."""

    @trace_function
    def __init__(
        self,
        config: AppConfig,
        platform_info: PlatformInfo,
        firewall_status: FirewallStatus,
        logger: logging.Logger,
        *,
        benchmark_result_path: Path | None = None,
        bootstrap_script_path: Path | None = None,
    ) -> None:
        """Store shared runtime services used during startup coordination.

        Args:
            config: Application configuration for discovery and runtime startup.
            platform_info: Platform facts discovered during bootstrap.
            firewall_status: Result of the startup firewall preparation step.
            logger: Logger used for supervisor lifecycle messages.
            benchmark_result_path: Path to ``result.json``. When provided, the
                supervisor reads it to learn which backends this host can run
                and validates ``--backend`` / ``--dual-purpose`` requests
                against that set.
            bootstrap_script_path: Path to ``bootstrap.py``, used when spawning
                a peer compute-node subprocess. Required when ``dual_purpose``
                or the default-compute peer-spawn flow is active.
        """
        self.config = config
        self.platform_info = platform_info
        self.firewall_status = firewall_status
        self.logger = logger
        self.state = RuntimeState.INIT
        self._shutdown_requested = False
        self.benchmark_result_path = benchmark_result_path
        self.bootstrap_script_path = bootstrap_script_path
        self.available_backends: frozenset[str] = (
            frozenset(load_usable_backends(benchmark_result_path))
            if benchmark_result_path is not None
            else frozenset()
        )
        self._peer_process: subprocess.Popen[bytes] | None = None
        self._peer_backend: str | None = None
        self._peer_node_name: str | None = None
        self._peer_watcher_thread: threading.Thread | None = None
        self._peer_heartbeat_thread: threading.Thread | None = None
        self._peer_heartbeat_listener: SupervisorHeartbeatListener | None = None
        self._peer_terminate_requested = False
        self._validate_backend_flags()

    def _validate_backend_flags(self) -> None:
        """Fail fast when ``--backend`` requests a backend the host cannot run.

        Use this during construction so an impossible request is surfaced before
        discovery, firewall, or runtime work begins.
        """
        pinned = self.config.pinned_backend
        if pinned is None:
            return
        if not self.available_backends:
            raise RuntimeError(
                f"--backend {pinned} was requested but no benchmark result was available; "
                f"re-run the local benchmark so the supervisor can verify backend support."
            )
        if pinned not in self.available_backends:
            raise RuntimeError(
                f"--backend {pinned} is not supported on this host; "
                f"result.json reports usable backends={sorted(self.available_backends)}."
            )

    def _best_available_gpu(self) -> str | None:
        """Return the highest-priority GPU backend this host can run, if any.

        Returns:
            A backend name such as ``cuda`` or ``metal``, or ``None`` when no
            GPU backend appears in the benchmark result.
        """
        for candidate in GPU_BACKEND_PRIORITY:
            if candidate in self.available_backends:
                return candidate
        return None

    def _plan_capacity(self, resolved_role: str) -> tuple[str | None, str | None]:
        """Decide this process's effective backend and whether to spawn a peer.

        Use this after discovery settles the role so occupied-backend tracking
        and peer-spawn decisions can be made from one authoritative place.

        Args:
            resolved_role: ``"main"`` when this process is running as main-node,
                or ``"compute"`` when it is running as a compute-node.

        Returns:
            ``(self_backend, peer_backend)`` where ``self_backend`` is the
            backend this process pins to (``None`` for main-node, since main
            does not run compute work directly) and ``peer_backend`` is the
            backend the supervisor should spawn a peer for (``None`` for no
            peer).
        """
        if resolved_role == "main":
            if not self.config.dual_purpose:
                return (None, None)
            best_gpu = self._best_available_gpu()
            if best_gpu is None:
                self.logger.warning(
                    "--dual-purpose was requested but no GPU backend is available on this host "
                    "(usable_backends=%s); running main-node only.",
                    sorted(self.available_backends) or ["none"],
                )
                return (None, None)
            return (None, best_gpu)

        pinned = self.config.pinned_backend
        if pinned is not None:
            return (pinned, None)

        best_gpu = self._best_available_gpu()
        if best_gpu is None:
            self.logger.info(
                "No GPU backend is available on this host; compute-node will run cpu only "
                "without a peer instance."
            )
            return (BACKEND_CPU, None)
        return (best_gpu, BACKEND_CPU)

    def _peer_command(self, peer_backend: str) -> list[str]:
        """Build the argv used to spawn a peer compute-node subprocess.

        Args:
            peer_backend: Backend name the peer should pin to.

        Returns:
            A python argv list suitable for ``subprocess.Popen``.
        """
        if self.bootstrap_script_path is None:
            raise RuntimeError(
                "bootstrap_script_path was not provided to Supervisor; "
                "cannot spawn a peer compute-node subprocess."
            )
        base_name = self.config.node_name or "node"
        arguments: list[str] = [
            sys.executable,
            self.bootstrap_script_path,
            "--role",
            "discover",
            "--backend",
            peer_backend,
            "--node-name",
            f"{base_name}-peer-{peer_backend}",
            "--multicast-group",
            self.config.multicast_group,
            "--udp-port",
            str(self.config.udp_port),
            "--tcp-port",
            str(self.config.tcp_port),
            "--data-plane-port",
            str(self.config.data_plane_port),
            "--no-manual-fallback",
            "--peer-process",
        ]
        if self.config.no_cli:
            arguments.append("--no-cli")
        return python_utf8_command(*arguments)

    def _peer_popen_kwargs(self) -> dict:
        """Return the ``Popen`` kwargs that shape the peer's console / IO state.

        Default mode gives the peer its own visible console window on Windows
        so operators and demos can see the peer's audit output directly
        (previously peers shared the parent's console and their lines got
        interleaved with the main-node's TRACE stream). ``--no-cli`` mode
        instead spawns the peer fully detached with ``DEVNULL`` IO, matching
        the parent's headless contract.
        """
        kwargs: dict = {}
        if self.config.no_cli:
            kwargs["stdin"] = subprocess.DEVNULL
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
            if sys.platform == "win32":
                detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                kwargs["creationflags"] = detached | no_window | new_group
            else:
                kwargs["start_new_session"] = True
            return kwargs

        if sys.platform == "win32":
            new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
            kwargs["creationflags"] = new_console
        return kwargs

    def _spawn_peer(self, peer_backend: str) -> None:
        """Start the peer compute-node subprocess and record its handle.

        Args:
            peer_backend: Backend name the peer should pin to.
        """
        command = self._peer_command(peer_backend)
        self.logger.info(
            "Spawning peer compute-node process pinned to backend=%s: %s",
            peer_backend,
            " ".join(command),
        )
        popen_kwargs = self._peer_popen_kwargs()
        # Build the heartbeat listener BEFORE spawn so the peer can connect
        # back on the first attempt. Failure to bind (e.g. ephemeral ports
        # exhausted) is not fatal — peer runs without heartbeat and the
        # exit-code watcher remains the fallback.
        heartbeat_listener: SupervisorHeartbeatListener | None = None
        try:
            heartbeat_listener = SupervisorHeartbeatListener()
        except OSError as exc:
            self.logger.warning(
                "Could not open peer heartbeat listener; peer will run without IPC liveness: %s",
                exc,
            )
        if heartbeat_listener is not None:
            child_env = {**os.environ, HEARTBEAT_PORT_ENV: str(heartbeat_listener.port)}
            popen_kwargs["env"] = child_env
        try:
            self._peer_process = subprocess.Popen(command, **popen_kwargs)
        except OSError as exc:
            self.logger.error(
                "Failed to spawn peer compute-node for backend=%s: %s",
                peer_backend,
                exc,
            )
            self._peer_process = None
            if heartbeat_listener is not None:
                heartbeat_listener.close()
            return
        self._peer_backend = peer_backend
        base_name = self.config.node_name or "node"
        self._peer_node_name = f"{base_name}-peer-{peer_backend}"
        self._peer_terminate_requested = False
        self._peer_heartbeat_listener = heartbeat_listener
        write_audit_event(
            f"spawned peer compute-node backend={peer_backend} pid={self._peer_process.pid}",
            logger=self.logger,
        )
        self._start_peer_watcher(self._peer_process, peer_backend)
        if heartbeat_listener is not None:
            self._start_peer_heartbeat_watcher(
                heartbeat_listener, self._peer_process, peer_backend
            )

    def _start_peer_watcher(
        self,
        process: subprocess.Popen[bytes],
        peer_backend: str,
    ) -> None:
        """Start a daemon thread that logs the cause of death of the peer.

        Why: when a peer subprocess dies mid-run (OOM-killer, segfault in a
        native runner, parent crash, etc.) the supervisor previously had no
        record — the peer's own console window closed and its diagnostics were
        lost. This watcher does NOT respawn; it only writes one audit/log line
        with the classified exit code so the operator can tell an OS-normal
        eviction from a runner bug.
        """
        thread = threading.Thread(
            target=self._watch_peer,
            args=(process, peer_backend),
            name=f"peer-watcher-{peer_backend}-{process.pid}",
            daemon=True,
        )
        self._peer_watcher_thread = thread
        thread.start()

    def _watch_peer(
        self,
        process: subprocess.Popen[bytes],
        peer_backend: str,
    ) -> None:
        """Block on ``process.wait`` and log a classified exit message.

        Distinguishes between "we asked it to terminate" (expected, INFO) and
        "it died on its own" (unexpected, WARNING) so a normal shutdown does
        not look like a crash in the logs.
        """
        try:
            returncode = process.wait()
        except Exception as exc:  # noqa: BLE001 - daemon thread must not propagate
            self.logger.error(
                "Peer watcher for backend=%s pid=%s failed while waiting: %s",
                peer_backend,
                process.pid,
                exc,
            )
            return
        cause = classify_exit_code(returncode)
        if self._peer_terminate_requested:
            self.logger.info(
                "Peer compute-node exited as requested: backend=%s pid=%s %s",
                peer_backend,
                process.pid,
                cause,
            )
            return
        message = (
            f"peer compute-node died unexpectedly backend={peer_backend} "
            f"pid={process.pid} returncode={returncode} cause=\"{cause}\" "
            f"(no respawn — observability only)"
        )
        self.logger.warning(message)
        write_audit_event(message, logger=self.logger, level=logging.WARNING)

    def _start_peer_heartbeat_watcher(
        self,
        listener: SupervisorHeartbeatListener,
        process: subprocess.Popen[bytes],
        peer_backend: str,
    ) -> None:
        """Spawn the daemon thread that reads heartbeats from the peer.

        The thread owns the ``accept`` so the main supervisor thread never
        blocks on peer startup. See :meth:`_watch_peer_heartbeat` for the
        detection semantics.
        """
        thread = threading.Thread(
            target=self._watch_peer_heartbeat,
            args=(listener, process, peer_backend),
            name=f"peer-heartbeat-{peer_backend}-{process.pid}",
            daemon=True,
        )
        self._peer_heartbeat_thread = thread
        thread.start()

    def _watch_peer_heartbeat(
        self,
        listener: SupervisorHeartbeatListener,
        process: subprocess.Popen[bytes],
        peer_backend: str,
    ) -> None:
        """Read heartbeat bytes from the peer; declare a hang on K misses.

        Called on the ``peer-heartbeat-*`` daemon thread. Declares the peer
        hung when both of the following hold at once:
          * ``HEARTBEAT_MISS_THRESHOLD`` consecutive reads timed out.
          * ``process.poll() is None`` (the OS process is still running).
        On a hang, dumps py-spy and terminates the peer. The watcher returns
        after firing, so re-entry is not possible on the same peer.
        """
        if not listener.accept(timeout=_PEER_HEARTBEAT_ACCEPT_TIMEOUT_SECONDS):
            self.logger.warning(
                "Peer compute-node backend=%s pid=%s never connected to heartbeat socket; "
                "liveness will rely on the exit-code watcher only.",
                peer_backend,
                process.pid,
            )
            return
        self.logger.info(
            "Peer heartbeat connected for backend=%s pid=%s", peer_backend, process.pid
        )
        miss_count = 0
        read_timeout = HEARTBEAT_INTERVAL_SECONDS * 2
        while not self._shutdown_requested and not self._peer_terminate_requested:
            status = listener.wait_for_heartbeat(read_timeout)
            if status == HEARTBEAT_OK:
                miss_count = 0
                continue
            if status == HEARTBEAT_CLOSED:
                # Peer closed its side of the socket. The peer-exit path runs
                # the socket.close() in the writer thread's ``finally``, so
                # this is the peer's "I am exiting" signal. Defer to the
                # exit-code watcher for logging; do NOT accumulate misses
                # (EOF arrives in microseconds and would trip the threshold
                # before poll() has even reaped the process).
                self.logger.debug(
                    "Peer heartbeat channel closed by peer (backend=%s pid=%s); "
                    "letting exit-code watcher report the cause.",
                    peer_backend,
                    process.pid,
                )
                return
            # status == HEARTBEAT_TIMEOUT
            if process.poll() is not None:
                return
            miss_count += 1
            self.logger.debug(
                "Peer heartbeat miss %d/%d for backend=%s pid=%s",
                miss_count,
                HEARTBEAT_MISS_THRESHOLD,
                peer_backend,
                process.pid,
            )
            if miss_count < HEARTBEAT_MISS_THRESHOLD:
                continue
            self._dump_and_terminate_hung_peer(
                process,
                f"{HEARTBEAT_MISS_THRESHOLD} consecutive heartbeat misses over IPC",
            )
            return

    def _dump_and_terminate_hung_peer(
        self,
        process: subprocess.Popen[bytes],
        reason: str,
    ) -> None:
        """Capture a py-spy stack from a still-running peer and terminate it."""
        self.logger.warning(
            "Peer compute-node %s is alive but hung (pid=%s, reason=%s); "
            "capturing py-spy stack before terminating.",
            self._peer_node_name,
            process.pid,
            reason,
        )
        stack_dump = dump_python_stack(process.pid)
        message = (
            f"hung peer stack dump backend={self._peer_backend} pid={process.pid} "
            f"node={self._peer_node_name} reason=\"{reason}\":\n{stack_dump}"
        )
        self.logger.warning(message)
        write_audit_event(message, logger=self.logger, level=logging.WARNING)
        self._terminate_peer()

    def _terminate_peer(self) -> None:
        """Best-effort terminate of the peer compute-node on shutdown.

        Use this from ``shutdown`` so peer subprocesses do not outlive the
        parent supervisor. The supervisor does not respawn dead peers, so this
        is the only lifecycle hook the peer gets.
        """
        process = self._peer_process
        if process is None:
            return
        if process.poll() is not None:
            self._peer_process = None
            return
        self._peer_terminate_requested = True
        self.logger.info(
            "Terminating peer compute-node pid=%s backend=%s",
            process.pid,
            self._peer_backend,
        )
        try:
            process.terminate()
            process.wait(timeout=_PEER_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Peer compute-node pid=%s did not exit within %.1fs; killing.",
                process.pid,
                _PEER_TERMINATE_TIMEOUT_SECONDS,
            )
            process.kill()
            try:
                process.wait(timeout=_PEER_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.logger.error(
                    "Peer compute-node pid=%s did not exit after kill.",
                    process.pid,
                )
        except OSError as exc:
            self.logger.warning("Failed to terminate peer compute-node: %s", exc)
        finally:
            self._peer_process = None
            listener = self._peer_heartbeat_listener
            self._peer_heartbeat_listener = None
            if listener is not None:
                listener.close()

    @trace_function
    def register_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers."""

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    @trace_function
    def _handle_signal(self, signum: int, _frame: object) -> None:
        """Handle termination signals."""

        self._shutdown_requested = True
        self.logger.info("Received signal %s, preparing shutdown.", signum)

    @trace_function
    def _set_state(self, state: RuntimeState) -> None:
        """Record and log one supervisor state transition."""
        self.state = state
        self.logger.info("Supervisor state -> %s", state.value)

    def _mac_seeded_pre_discovery_delay(self) -> float:
        """Return a MAC-derived deterministic stagger before discovery starts.

        Why: two machines booting at the same moment both finish discovery at
        the same moment and both promote to main (split brain). MAC-derived
        slotting gives each machine a different start offset so the later
        starter has time to hear the earlier starter's ANNOUNCE.
        """
        slot = uuid.getnode() % _PRE_DISCOVERY_STAGGER_SLOTS
        return slot * (_PRE_DISCOVERY_STAGGER_MAX_SECONDS / _PRE_DISCOVERY_STAGGER_SLOTS)

    def _next_discovery_retry_delay(self, attempt: int) -> float:
        """Return a jittered retry delay for the next discovery attempt.

        Args:
            attempt: 1-based discovery attempt that just failed.

        Returns:
            A randomized delay in seconds derived from the configured base delay.
        """
        del attempt
        base_delay = max(self.config.discovery_retry_delay, 0.0)
        if base_delay <= 0:
            return 0.0
        return random.uniform(base_delay * 0.5, base_delay * 1.5)

    @trace_function
    def _discover_with_retries(self) -> DiscoveryResult:
        """Try discovery several times before promoting self to the main node."""

        last_result = DiscoveryResult(success=False, message="Discovery did not run.")
        write_audit_event("discovering main node", stdout=True, logger=self.logger)

        for attempt in range(1, self.config.discovery_attempts + 1):
            if self._shutdown_requested:
                return DiscoveryResult(success=False, message="Shutdown requested during discovery.")

            self.logger.info(
                "Discovery attempt %s/%s",
                attempt,
                self.config.discovery_attempts,
            )
            last_result = run_pairing(self.config)
            if last_result.success:
                return last_result

            self.logger.warning(
                "Discovery attempt %s failed: %s",
                attempt,
                last_result.message,
            )
            if attempt < self.config.discovery_attempts:
                retry_delay = self._next_discovery_retry_delay(attempt)
                self.logger.info(
                    "Waiting %.2fs before discovery retry %s/%s",
                    retry_delay,
                    attempt + 1,
                    self.config.discovery_attempts,
                )
                time.sleep(retry_delay)

        return last_result

    @trace_function
    def _promote_to_main_node(self) -> DiscoveryResult:
        """Become the main node and continue listening for multicast traffic."""

        write_audit_event("promoting self to main node", stdout=True, logger=self.logger)
        self._set_state(RuntimeState.MAIN_NODE)
        self.logger = rebind_logging_role("main")
        write_audit_event("promoting self to main node", logger=self.logger)
        _, peer_backend = self._plan_capacity("main")
        if peer_backend is not None and not self._shutdown_requested:
            self._spawn_peer(peer_backend)
        runtime = MainNodeRuntime(
            config=self.config,
            logger=self.logger,
            should_stop=lambda: self._shutdown_requested,
        )
        return runtime.run()

    @trace_function
    def _join_main_node(self, result: DiscoveryResult) -> DiscoveryResult:
        """Use a discovered main-node address to enter compute-node runtime."""

        if not result.peer_address or not result.peer_port:
            return DiscoveryResult(
                success=False,
                message="Discovery succeeded but did not include a main-node TCP endpoint.",
            )

        write_audit_event(
            f"joining discovered main node {result.peer_address}:{result.peer_port} as compute node",
            stdout=True,
            logger=self.logger,
        )
        self._set_state(RuntimeState.COMPUTE_NODE)
        self.logger = rebind_logging_role("worker")
        self_backend, peer_backend = self._plan_capacity("compute")
        if self_backend is not None:
            self.config.pinned_backend = self_backend
        if peer_backend is not None and not self._shutdown_requested:
            self._spawn_peer(peer_backend)
        runtime = ComputeNodeRuntime(
            config=self.config,
            main_node_host=result.peer_address,
            main_node_port=result.peer_port,
            logger=self.logger,
            should_stop=lambda: self._shutdown_requested,
        )
        return runtime.run()

    @trace_function
    def run(self) -> DiscoveryResult:
        """Run discovery and optional manual fallback."""

        self.register_signal_handlers()
        try:
            # Discover-mode compute nodes try several times to find an existing
            # main node. If no peer responds, the process promotes itself and
            # starts listening for future compute nodes on the multicast group.
            if self.config.role == "discover":
                self._set_state(RuntimeState.DISCOVERY)
                pre_delay = self._mac_seeded_pre_discovery_delay()
                if pre_delay > 0 and not self._shutdown_requested:
                    self.logger.info(
                        "MAC-seeded pre-discovery stagger: sleeping %.3fs",
                        pre_delay,
                    )
                    time.sleep(pre_delay)
                result = self._discover_with_retries()
                if result.success and not self._shutdown_requested:
                    return self._join_main_node(result)
                if self._shutdown_requested:
                    return result

                self.logger.warning("Discovery failed after retry flow: %s", result.message)

                promoted_result = self._promote_to_main_node()
                if promoted_result.success or self._shutdown_requested:
                    return promoted_result
                self.logger.warning("Main-node promotion failed: %s", promoted_result.message)

                if self.config.enable_manual_fallback:
                    self._set_state(RuntimeState.MANUAL_INPUT)
                    manual_result = prompt_manual_address(self.config.tcp_port)
                    if manual_result.success and not self._shutdown_requested:
                        return self._join_main_node(manual_result)
                    return manual_result

                return promoted_result

            return self._promote_to_main_node()
        finally:
            self._set_state(RuntimeState.IDLE)

    @trace_function
    def shutdown(self) -> FirewallStatus:
        """Best-effort firewall cleanup on exit."""

        self._set_state(RuntimeState.SHUTDOWN)
        self._terminate_peer()
        if self.config.peer_process:
            status = FirewallStatus(
                supported=True,
                applied=False,
                needs_admin=False,
                backend=self.platform_info.platform_name,
                message="firewall cleanup delegated to spawning supervisor (peer process)",
            )
            self.logger.info("Firewall cleanup: %s", status.message)
            return status
        status = cleanup_rules(self.platform_info, self.config.udp_port)
        self.logger.info("Firewall cleanup: %s", status.message)
        return status

