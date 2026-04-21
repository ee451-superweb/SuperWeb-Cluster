"""Minimal supervisor for superweb-cluster Sprint 1."""

from __future__ import annotations

import logging
import random
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

_PRE_DISCOVERY_STAGGER_MAX_SECONDS = 1.5
_PRE_DISCOVERY_STAGGER_SLOTS = 1024
_PEER_TERMINATE_TIMEOUT_SECONDS = 5.0

from adapters.firewall import cleanup_rules
from adapters.audit_log import write_audit_event
from adapters.process import python_utf8_command
from app.capacity import load_usable_backends
from app.constants import (
    BACKEND_CPU,
    GPU_BACKEND_PRIORITY,
)
from common.state import RuntimeState
from common.types import DiscoveryResult, FirewallStatus, PlatformInfo
from compute_node.runtime import ComputeNodeRuntime
from app.config import AppConfig
from app.logging_setup import rebind_logging_role
from discovery.fallback import prompt_manual_address
from discovery.pairing import run_pairing
from main_node.runtime import MainNodeRuntime
from app.trace_utils import trace_function


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
        command = python_utf8_command(
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
        )
        return command

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
        try:
            self._peer_process = subprocess.Popen(command)
        except OSError as exc:
            self.logger.error(
                "Failed to spawn peer compute-node for backend=%s: %s",
                peer_backend,
                exc,
            )
            self._peer_process = None
            return
        self._peer_backend = peer_backend
        write_audit_event(
            f"spawned peer compute-node backend={peer_backend} pid={self._peer_process.pid}",
            logger=self.logger,
        )

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
        status = cleanup_rules(self.platform_info, self.config.udp_port)
        self.logger.info("Firewall cleanup: %s", status.message)
        return status

