"""Program entry point for the kickoff version."""

from __future__ import annotations

import argparse
import sys

from adapters.firewall import ensure_rules
from adapters.platform import detect_os, relaunch_as_admin
from config import AppConfig
from constants import (
    APP_NAME,
    DEFAULT_NODE_NAME,
    DEFAULT_DISCOVERY_ATTEMPTS,
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_DISCOVERY_RETRY_DELAY,
    DEFAULT_DISCOVERY_TIMEOUT,
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_TCP_PORT,
    HOME_COMPUTER_NAME,
    HOME_SCHEDULER_NAME,
)
from logging_setup import configure_logging
from supervisor import Supervisor
from trace_utils import trace_function


@trace_function
def build_parser() -> argparse.ArgumentParser:
    """Build the kickoff CLI."""

    parser = argparse.ArgumentParser(description=f"{APP_NAME} kickoff bootstrap.")
    parser.add_argument("--role", choices=("discover", "announce"), default="discover")
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument("--multicast-group", default=DEFAULT_MULTICAST_GROUP)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_DISCOVERY_PORT)
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT)
    parser.add_argument(
        "--discover-attempts",
        type=int,
        default=DEFAULT_DISCOVERY_ATTEMPTS,
        help="How many discovery attempts to make before promoting self to the home scheduler.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_DISCOVERY_RETRY_DELAY,
        help="Seconds to wait between discovery attempts.",
    )
    parser.add_argument(
        "--manual-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable manual input fallback when discovery fails.",
    )
    parser.add_argument(
        "--elevate-if-needed",
        action="store_true",
        help="On Windows, relaunch with administrator privileges when needed.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


@trace_function
def build_config(args: argparse.Namespace) -> AppConfig:
    """Convert CLI arguments into an AppConfig."""

    return AppConfig(
        role=args.role,
        node_name=(
            HOME_SCHEDULER_NAME
            if args.node_name == DEFAULT_NODE_NAME and args.role == "announce"
            else HOME_COMPUTER_NAME
            if args.node_name == DEFAULT_NODE_NAME
            else args.node_name
        ),
        multicast_group=args.multicast_group,
        udp_port=args.udp_port,
        tcp_port=args.tcp_port,
        discovery_timeout=args.timeout,
        discovery_attempts=args.discover_attempts,
        discovery_retry_delay=args.retry_delay,
        enable_manual_fallback=args.manual_fallback,
    )


@trace_function
def main(argv: list[str] | None = None) -> int:
    """Run bootstrap and return a process exit code."""

    # Parse CLI arguments first so the rest of startup can use a single
    # normalized configuration object.
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(verbose=args.verbose)

    # Platform detection happens before firewall setup so we can route into the
    # correct adapter and decide whether elevation is even meaningful.
    platform_info = detect_os()
    logger.info(
        "Platform detected: %s (system=%s, release=%s, admin=%s, wsl=%s)",
        platform_info.platform_name,
        platform_info.system,
        platform_info.release,
        platform_info.is_admin,
        platform_info.is_wsl,
    )

    # Windows elevation is only attempted when the user explicitly asks for it.
    if args.elevate_if_needed and platform_info.can_elevate and relaunch_as_admin():
        logger.info("Relaunched with administrator privileges.")
        return 0

    config = build_config(args)
    # Firewall work is intentionally limited to discovery-phase UDP exposure.
    firewall_status = ensure_rules(platform_info, config.udp_port)
    logger.info("Firewall setup: %s", firewall_status.message)

    supervisor = Supervisor(
        config=config,
        platform_info=platform_info,
        firewall_status=firewall_status,
        logger=logger,
    )

    try:
        # The supervisor owns the rest of the kickoff lifecycle, including
        # discovery, fallback, and best-effort shutdown.
        result = supervisor.run()
    finally:
        supervisor.shutdown()

    if result.success:
        logger.info(
            "Final result: source=%s peer=%s:%s message=%s",
            result.source,
            result.peer_address,
            result.peer_port,
            result.message,
        )
        return 0

    logger.error("Final result: %s", result.message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
