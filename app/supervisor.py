"""Minimal supervisor for superweb-cluster Sprint 1."""

from __future__ import annotations

import logging
import random
import signal
import time
import uuid

_PRE_DISCOVERY_STAGGER_MAX_SECONDS = 1.5
_PRE_DISCOVERY_STAGGER_SLOTS = 1024

from adapters.firewall import cleanup_rules
from adapters.audit_log import write_audit_event
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
    ) -> None:
        """Store shared runtime services used during startup coordination.

        Args:
            config: Application configuration for discovery and runtime startup.
            platform_info: Platform facts discovered during bootstrap.
            firewall_status: Result of the startup firewall preparation step.
            logger: Logger used for supervisor lifecycle messages.
        """
        self.config = config
        self.platform_info = platform_info
        self.firewall_status = firewall_status
        self.logger = logger
        self.state = RuntimeState.INIT
        self._shutdown_requested = False

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
        status = cleanup_rules(self.platform_info, self.config.udp_port)
        self.logger.info("Firewall cleanup: %s", status.message)
        return status

