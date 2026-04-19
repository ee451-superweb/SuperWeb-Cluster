"""Top-level bootstrap entry point for superweb-cluster."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from adapters.firewall import ensure_rules
from adapters.platform import detect_os, relaunch_as_admin
from adapters.audit_log import write_audit_event
from app.config import AppConfig
from app.constants import (
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
from app.logging_setup import archive_existing_logs, cleanse_existing_logs, configure_logging
from app.trace_utils import trace_function
from setup import REQUIREMENTS_STAMP_PATH, active_python_path, inspect_project_environment, project_python_path

PROJECT_ROOT = Path(__file__).resolve().parent
COMPUTE_NODE_DIR = PROJECT_ROOT / "compute_node"
BENCHMARK_DIR = COMPUTE_NODE_DIR / "performance_metrics"
BENCHMARK_SCRIPT_PATH = BENCHMARK_DIR / "benchmark.py"
BENCHMARK_RESULT_PATH = BENCHMARK_DIR / "result.json"
INPUT_MATRIX_DIR = COMPUTE_NODE_DIR / "input_matrix"
INPUT_MATRIX_SCRIPT_PATH = INPUT_MATRIX_DIR / "generate.py"
INTERRUPTED_EXIT_CODE = 130


def _display_machine_label(machine: str) -> str:
    """Render one short operator-facing architecture label."""

    normalized = machine.strip().lower()
    if normalized in {"amd64", "x86_64", "x64"}:
        return "x64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    return normalized or "unknown"


def _platform_bootstrap_summary(platform_info) -> str:
    """Render one concise platform summary for startup audit logs."""

    return (
        f"os details: {platform_info.platform_name}, "
        f"{platform_info.release}, "
        f"{_display_machine_label(platform_info.machine)}, "
        f"{platform_info.hostname}"
    )


def _runtime_relaunch_argv(argv: list[str]) -> list[str]:
    """Build the bootstrap argv used when relaunching into the project venv.

    The relaunched process should default to the Windows elevation check so a
    system-Python bootstrap and the final venv-hosted runtime behave the same.

    Args:
        argv: Original bootstrap argument vector excluding the program path.

    Returns:
        A normalized argument list for the relaunched bootstrap process.
    """

    relaunch_argv = list(argv)
    if "--elevate-if-needed" not in relaunch_argv:
        relaunch_argv.append("--elevate-if-needed")
    return relaunch_argv


def _input_matrix_command(*, force_regenerate: bool = False) -> list[str]:
    """Build the local dataset-generation command using the current Python executable."""

    command = [str(active_python_path()), str(INPUT_MATRIX_SCRIPT_PATH), "--method", "all"]
    if force_regenerate:
        command.append("--force")
    return command


def _benchmark_command(*, force_rebuild: bool = False) -> list[str]:
    """Build the local benchmark command using the current Python executable."""

    command = [str(active_python_path()), str(BENCHMARK_SCRIPT_PATH), "--method", "all"]
    if force_rebuild:
        command.append("--rebuild")
    return command


def _setup_command() -> list[str]:
    """Build the setup command used to prepare the project venv and dependencies."""

    return [sys.executable, str(PROJECT_ROOT / "setup.py")]


def _run_streaming_command(
    command: list[str],
    *,
    cwd: Path,
    logger,
    log_level: int = logging.INFO,
) -> int:
    """Run one subprocess while mirroring stdout/stderr to stdout and logger.

    Use this for long-running bootstrap helpers such as dataset generation and
    benchmarking so operators can watch progress live while the same text is
    also persisted in the bootstrap session log.
    """

    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for raw_line in process.stdout:
            print(raw_line, end="", flush=True)
            message = raw_line.rstrip()
            if message:
                logger.log(log_level, message)
        return int(process.wait())


def _run_passthrough_command(
    command: list[str],
    *,
    cwd: Path,
) -> int:
    """Run one subprocess with direct console stdout/stderr passthrough.

    Use this for helpers that render interactive progress bars such as tqdm.
    Keeping the child process attached to the parent's console preserves the
    in-place progress updates that would otherwise degrade into many lines when
    routed through a pipe.
    """

    result = subprocess.run(
        command,
        check=False,
        cwd=cwd,
    )
    return int(result.returncode)


def _display_project_path(path: Path) -> str:
    """Render a project-local path for bootstrap log messages."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _apply_log_start_mode(mode: str) -> tuple[int, str] | None:
    """Apply one startup log policy before the current session log is created."""

    normalized_mode = mode.strip().lower()
    if normalized_mode == "normal":
        return None

    try:
        if normalized_mode == "clean":
            archive_path, archived_count = archive_existing_logs()
            if archive_path is None:
                return logging.INFO, "Log start mode clean found no previous loose log files to archive."
            return (
                logging.INFO,
                "Log start mode clean archived "
                f"{archived_count} previous log files into {_display_project_path(archive_path)}.",
            )

        removed_count = cleanse_existing_logs()
        return logging.INFO, f"Log start mode cleanse removed {removed_count} previous log artifacts."
    except OSError as exc:
        return logging.WARNING, f"Log start mode {normalized_mode} failed: {exc}"


def _validate_compute_benchmark_assets(logger, result_path: Path | None = None) -> None:
    """Validate that compute startup has refresh-ready datasets and runners.

    Args:
        logger: Bootstrap logger used to emit startup progress messages.
        result_path: Optional benchmark result path to validate instead of the default.

    Returns:
        ``None`` when idle refresh can reuse the persisted benchmark output safely.
    """
    from compute_node.performance_refresh import validate_idle_refresh_requirements

    resolved_result_path = BENCHMARK_RESULT_PATH if result_path is None else result_path
    logger.info(
        "Benchmark validation 0%%: starting startup checks for %s.",
        _display_project_path(resolved_result_path),
    )

    def log_progress(step: int, total_steps: int, description: str) -> None:
        percent = int(round((step / total_steps) * 100))
        logger.info(
            "Benchmark validation %s%% (%s/%s): %s",
            percent,
            step,
            total_steps,
            description,
        )

    validate_idle_refresh_requirements(
        resolved_result_path,
        progress_callback=log_progress,
    )
    logger.info("Benchmark validation 100%%: startup checks passed.")


@trace_function
def ensure_bootstrap_runtime_environment(logger, argv: list[str]) -> int | None:
    """Ensure bootstrap runs with the prepared project venv.

    Returns:
        `None` when bootstrap can continue locally.
        An exit code when bootstrap should stop or after it relaunches itself.
    """

    status = inspect_project_environment()
    logger.info(
        "Bootstrap interpreter check: current=%s project=%s using_project_python=%s ready=%s",
        sys.executable,
        project_python_path(),
        status.using_project_python,
        status.ready,
    )
    if not status.ready:
        setup_command = _setup_command()
        logger.info(
            "Project Python environment is not ready. Running setup now: %s",
            " ".join(setup_command),
        )
        if not status.venv_exists:
            logger.info("Missing local virtual environment at %s.", _display_project_path(project_python_path()))
        if not status.requirements_current:
            logger.info(
                "Project dependencies are missing or out of date for %s.",
                _display_project_path(REQUIREMENTS_STAMP_PATH),
            )
        try:
            setup_result = subprocess.run(
                setup_command,
                check=False,
                cwd=PROJECT_ROOT,
            )
        except OSError as exc:
            logger.error("Failed to run setup.py before bootstrap: %s", exc)
            return 1
        if setup_result.returncode != 0:
            logger.error("setup.py failed with exit code %s.", setup_result.returncode)
            return int(setup_result.returncode or 1)
        status = inspect_project_environment()
        if not status.ready:
            logger.error("Project Python environment is still not ready after setup.py completed.")
            if not status.venv_exists:
                logger.error("Missing local virtual environment at %s.", _display_project_path(project_python_path()))
            if not status.requirements_current:
                logger.error("Project dependencies are missing or out of date for %s.", _display_project_path(REQUIREMENTS_STAMP_PATH))
            return 1

    if status.using_project_python:
        return None

    relaunch_command = [
        str(project_python_path()),
        str(PROJECT_ROOT / "bootstrap.py"),
        *_runtime_relaunch_argv(argv),
    ]
    logger.info(
        "Relaunching bootstrap with the project virtual environment: %s",
        " ".join(relaunch_command),
    )
    try:
        result = subprocess.run(
            relaunch_command,
            check=False,
            cwd=PROJECT_ROOT,
        )
    except OSError as exc:
        logger.error("Failed to relaunch bootstrap with the project virtual environment: %s", exc)
        return 1
    return int(result.returncode)


def _load_supervisor_class():
    """Import Supervisor only after the runtime environment is ready."""

    from app.supervisor import Supervisor

    return Supervisor


@trace_function
def build_parser() -> argparse.ArgumentParser:
    """Build the kickoff CLI."""

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} bootstrap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--role",
        choices=("discover", "announce"),
        default="discover",
        help="Startup role. Use discover to look for an existing main node, or announce to start directly as the main node.",
    )
    parser.add_argument(
        "--node-name",
        default=DEFAULT_NODE_NAME,
        help="Cluster node label to advertise. The default resolves to the built-in main or compute role name.",
    )
    parser.add_argument(
        "--multicast-group",
        default=DEFAULT_MULTICAST_GROUP,
        help="IPv4 multicast group used for discovery traffic.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=DEFAULT_DISCOVERY_PORT,
        help="UDP discovery port used for multicast announce and discover traffic.",
    )
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=DEFAULT_TCP_PORT,
        help="TCP port used by the main-node runtime server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_DISCOVERY_TIMEOUT,
        help="Seconds to wait for each discovery reply before trying again.",
    )
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
    parser.add_argument(
        "--retest",
        action="store_true",
        help="Regenerate input matrices and rerun the initial compute benchmark before startup.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rerun the initial compute benchmark and force benchmark runner binaries to rebuild before startup without regenerating input matrices.",
    )
    parser.add_argument(
        "--log-start-mode",
        choices=("normal", "clean", "cleanse"),
        default="normal",
        help="How bootstrap should treat older log files before opening a new session log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging, including DEBUG-level bootstrap output.",
    )
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
@trace_function
def ensure_compute_node_benchmark_ready(
    logger,
    *,
    force_retest: bool = False,
    force_rebuild: bool = False,
) -> bool:
    """Make sure a local compute benchmark result exists before bootstrap continues."""

    if force_retest:
        logger.warning(
            "Bootstrap retest requested. Regenerating compute input matrices now.",
        )
        logger.info("Input-matrix command: %s", " ".join(_input_matrix_command(force_regenerate=True)))
        try:
            return_code = _run_passthrough_command(
                _input_matrix_command(force_regenerate=True),
                cwd=PROJECT_ROOT,
            )
        except OSError as exc:
            logger.error("Failed to regenerate compute input matrices automatically: %s", exc)
            return False
        if return_code != 0:
            logger.error(
                "Input-matrix generation exited with code %s during bootstrap retest.",
                return_code,
            )
            return False

    benchmark_needs_rebuild = force_retest or force_rebuild or not BENCHMARK_RESULT_PATH.exists()
    if BENCHMARK_RESULT_PATH.exists() and not (force_retest or force_rebuild):
        try:
            _validate_compute_benchmark_assets(logger, BENCHMARK_RESULT_PATH)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Existing compute benchmark assets are not refresh-ready: %s. Rebuilding the local benchmark now.",
                exc,
            )
            benchmark_needs_rebuild = True
        else:
            return True

    if force_rebuild:
        logger.warning(
            "Bootstrap rebuild requested. Running the local benchmark now and forcing backend rebuilds for %s.",
            _display_project_path(BENCHMARK_RESULT_PATH),
        )
    elif force_retest:
        logger.warning(
            "Bootstrap retest requested. Running the local benchmark now for %s.",
            _display_project_path(BENCHMARK_RESULT_PATH),
        )
    elif benchmark_needs_rebuild:
        logger.warning(
            "Compute benchmark assets are missing or stale at %s. Running the local benchmark now.",
            _display_project_path(BENCHMARK_RESULT_PATH),
        )
    logger.info("Benchmark command: %s", " ".join(_benchmark_command(force_rebuild=force_rebuild)))

    try:
        return_code = _run_streaming_command(
            _benchmark_command(force_rebuild=force_rebuild),
            cwd=PROJECT_ROOT,
            logger=logger,
        )
    except OSError as exc:
        logger.error("Failed to run compute benchmark automatically: %s", exc)
        return False
    if return_code != 0:
        logger.error(
            "Compute benchmark exited with code %s during bootstrap startup.",
            return_code,
        )
        return False

    if not BENCHMARK_RESULT_PATH.exists():
        logger.error(
            "Benchmark finished but %s was still not created.",
            _display_project_path(BENCHMARK_RESULT_PATH),
        )
        return False

    try:
        _validate_compute_benchmark_assets(logger, BENCHMARK_RESULT_PATH)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("Benchmark finished but compute assets are still not refresh-ready: %s", exc)
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
    log_start_summary = _apply_log_start_mode(args.log_start_mode)
    logger = configure_logging(
        verbose=args.verbose,
        role="main" if args.role == "announce" else "bootstrap",
    )
    if log_start_summary is not None:
        level, message = log_start_summary
        logger.log(level, message)
    supervisor = None
    try:
        effective_argv = list(argv) if argv is not None else sys.argv[1:]
        relaunch_result = ensure_bootstrap_runtime_environment(logger, effective_argv)
        if relaunch_result is not None:
            return relaunch_result

        if not ensure_compute_node_benchmark_ready(
            logger,
            force_retest=args.retest,
            force_rebuild=args.rebuild,
        ):
            return 1

        # Import runtime-heavy modules only after the project Python environment
        # check has had a chance to relaunch into `.venv`.
        Supervisor = _load_supervisor_class()

        # Platform detection happens before firewall setup so we can route into the
        # correct adapter and decide whether elevation is even meaningful.
        platform_info = detect_os()
        write_audit_event(
            _platform_bootstrap_summary(platform_info),
            stdout=True,
            logger=logger,
        )
        logger.info(
            "Platform detected: %s (system=%s, release=%s, machine=%s, hostname=%s, admin=%s, wsl=%s)",
            platform_info.platform_name,
            platform_info.system,
            platform_info.release,
            platform_info.machine,
            platform_info.hostname,
            platform_info.is_admin,
            platform_info.is_wsl,
        )

        # Windows elevation is attempted when explicitly requested, and the
        # self-relaunched venv process opts into the same check by default.
        if args.elevate_if_needed and platform_info.can_elevate and relaunch_as_admin():
            logger.info("Relaunched with administrator privileges.")
            return 0

        config = build_config(args)
        # Firewall work is intentionally limited to discovery-phase UDP exposure.
        write_audit_event("setting up firewall", stdout=True, logger=logger)
        firewall_status = ensure_rules(platform_info, config.udp_port)
        logger.info("Firewall setup: %s", firewall_status.message)

        supervisor = Supervisor(
            config=config,
            platform_info=platform_info,
            firewall_status=firewall_status,
            logger=logger,
        )

        # The supervisor owns the rest of the kickoff lifecycle, including
        # discovery, fallback, and best-effort shutdown.
        result = supervisor.run()
    except KeyboardInterrupt:
        logger.warning("Bootstrap interrupted by Ctrl+C.")
        return INTERRUPTED_EXIT_CODE
    finally:
        if supervisor is not None:
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
