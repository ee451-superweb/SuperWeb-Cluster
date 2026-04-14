"""Minimal supervisor for superweb-cluster Sprint 1."""

from __future__ import annotations

import logging
import signal
import time

from adapters.firewall import cleanup_rules
from common.state import RuntimeState
from common.types import DiscoveryResult, FirewallStatus, PlatformInfo
from compute_node.runtime import ComputeNodeRuntime
from app.config import AppConfig
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
        self.state = state
        self.logger.info("Supervisor state -> %s", state.value)

    @trace_function
    def _discover_with_retries(self) -> DiscoveryResult:
        """Try discovery several times before promoting self to the main node."""

        last_result = DiscoveryResult(success=False, message="Discovery did not run.")

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
                time.sleep(self.config.discovery_retry_delay)

        return last_result

    @trace_function
    def _promote_to_main_node(self) -> DiscoveryResult:
        """Become the main node and continue listening for multicast traffic."""

        self._set_state(RuntimeState.MAIN_NODE)
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

        self._set_state(RuntimeState.COMPUTE_NODE)
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

