"""Top-level bootstrap entry point for superweb-cluster."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from adapters.firewall import ensure_rules
from adapters.platform import detect_os, relaunch_as_admin
from config import AppConfig
from constants import (
    APP_NAME,
    COMPUTE_NODE_NAME,
    DEFAULT_NODE_NAME,
    DEFAULT_DISCOVERY_ATTEMPTS,
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_DISCOVERY_RETRY_DELAY,
    DEFAULT_DISCOVERY_TIMEOUT,
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_TCP_PORT,
    MAIN_NODE_NAME,
)
from logging_setup import configure_logging
from supervisor import Supervisor
from trace_utils import trace_function

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP_PATH = VENV_DIR / ".requirements.sha256"
COMPUTE_NODE_DIR = PROJECT_ROOT / "compute_node"
BENCHMARK_DIR = COMPUTE_NODE_DIR / "performance_metrics"
BENCHMARK_SCRIPT_PATH = BENCHMARK_DIR / "benchmark.py"
BENCHMARK_RESULT_PATH = BENCHMARK_DIR / "result.json"


def _project_python_path() -> Path:
    """Return the Python executable inside the project's virtual environment."""

    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _active_python_path() -> Path:
    """Prefer the project venv interpreter when it already exists."""

    project_python = _project_python_path()
    if project_python.exists():
        return project_python
    return Path(sys.executable)


def _benchmark_command() -> list[str]:
    """Build the local benchmark command using the current Python executable."""

    return [str(_active_python_path()), str(BENCHMARK_SCRIPT_PATH)]


def _requirements_hash() -> str:
    """Fingerprint `requirements.txt` so installs only rerun when it changes."""

    if not REQUIREMENTS_PATH.exists():
        return ""
    return hashlib.sha256(REQUIREMENTS_PATH.read_bytes()).hexdigest()


@trace_function
def build_parser() -> argparse.ArgumentParser:
    """Build the kickoff CLI."""

    parser = argparse.ArgumentParser(description=f"{APP_NAME} bootstrap.")
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
        help="How many discovery attempts to make before promoting self to the main node.",
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
            MAIN_NODE_NAME
            if args.node_name == DEFAULT_NODE_NAME and args.role == "announce"
            else COMPUTE_NODE_NAME
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


def _display_project_path(path: Path) -> str:
    """Render a project-local path for user-facing log messages."""

    return path.relative_to(PROJECT_ROOT).as_posix()


@trace_function
def ensure_project_python_environment(logger) -> bool:
    """Create `.venv` and install `requirements.txt` when needed."""

    try:
        project_python = _project_python_path()
        if not project_python.exists():
            logger.info("Creating project virtual environment at %s.", _display_project_path(VENV_DIR))
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                check=True,
                cwd=PROJECT_ROOT,
            )

        requirements_hash = _requirements_hash()
        installed_hash = REQUIREMENTS_STAMP_PATH.read_text(encoding="utf-8").strip() if REQUIREMENTS_STAMP_PATH.exists() else ""
        if requirements_hash and requirements_hash != installed_hash:
            logger.info(
                "Installing project requirements from %s using %s.",
                _display_project_path(REQUIREMENTS_PATH),
                _display_project_path(project_python),
            )
            subprocess.run(
                [str(project_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
                check=True,
                cwd=PROJECT_ROOT,
            )
            REQUIREMENTS_STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
            REQUIREMENTS_STAMP_PATH.write_text(requirements_hash, encoding="utf-8")
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Failed to prepare project virtual environment: %s", exc)
        return False


@trace_function
def ensure_compute_node_benchmark_ready(logger) -> bool:
    """Make sure a local compute benchmark result exists before bootstrap continues."""

    if BENCHMARK_RESULT_PATH.exists():
        return True

    logger.warning(
        "Missing compute benchmark result at %s. Running the local benchmark now.",
        _display_project_path(BENCHMARK_RESULT_PATH),
    )
    logger.info("Benchmark command: %s", " ".join(_benchmark_command()))

    try:
        subprocess.run(
            _benchmark_command(),
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Failed to run compute benchmark automatically: %s", exc)
        return False

    if not BENCHMARK_RESULT_PATH.exists():
        logger.error(
            "Benchmark finished but %s was still not created.",
            _display_project_path(BENCHMARK_RESULT_PATH),
        )
        return False

    logger.info(
        "Benchmark completed and wrote %s.",
        _display_project_path(BENCHMARK_RESULT_PATH),
    )
    return True


@trace_function
def main(argv: list[str] | None = None) -> int:
    """Run bootstrap and return a process exit code."""

    # Parse CLI arguments first so the rest of startup can use a single
    # normalized configuration object.
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logging(verbose=args.verbose)

    if not ensure_project_python_environment(logger):
        return 1

    if not ensure_compute_node_benchmark_ready(logger):
        return 1

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
