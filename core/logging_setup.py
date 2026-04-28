"""Role-aware logging configuration helpers."""

from __future__ import annotations

import logging
import zipfile
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.constants import (
    DEFAULT_LOG_FILE_BACKUP_COUNT,
    DEFAULT_LOG_FILE_MAX_BYTES,
    LOGGER_NAME,
    LOG_DIRECTORY_NAME,
)
from core.tracing import trace_function

_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_CURRENT_VERBOSE = False
_CURRENT_ROLE = "bootstrap"


def _log_directory() -> Path:
    """Return the project-local directory used for rotated runtime logs."""

    return Path(__file__).resolve().parents[1] / LOG_DIRECTORY_NAME


def _log_path_for_role(role: str) -> Path:
    """Return the file path used for one role's current-session log file."""

    normalized_role = role.strip().lower() or "bootstrap"
    return _log_directory() / f"{normalized_role}-{_SESSION_TIMESTAMP}.txt"


def _is_text_log_file(path: Path) -> bool:
    """Return whether a path is one runtime text log or its rotated shards."""

    return path.is_file() and ".txt" in path.suffixes


def _is_log_archive(path: Path) -> bool:
    """Return whether a path is one bootstrap-created log archive."""

    return path.is_file() and path.suffix.lower() == ".zip" and path.name.startswith("logs-archive-")


def _iter_log_artifacts(*, include_archives: bool) -> list[Path]:
    """Return existing log artifacts in the project log directory."""

    log_dir = _log_directory()
    if not log_dir.exists():
        return []
    artifacts = []
    for path in sorted(log_dir.iterdir()):
        if _is_text_log_file(path):
            artifacts.append(path)
            continue
        if include_archives and _is_log_archive(path):
            artifacts.append(path)
    return artifacts


def _archive_path_for_current_session() -> Path:
    """Return one unique archive path for this bootstrap session."""

    log_dir = _log_directory()
    candidate = log_dir / f"logs-archive-{_SESSION_TIMESTAMP}.zip"
    suffix = 1
    while candidate.exists():
        candidate = log_dir / f"logs-archive-{_SESSION_TIMESTAMP}-{suffix}.zip"
        suffix += 1
    return candidate


def _archive_compression() -> int:
    """Return the preferred zip compression algorithm for log archives."""

    return getattr(zipfile, "ZIP_ZSTANDARD", zipfile.ZIP_DEFLATED)


def _reset_root_handlers(root_logger: logging.Logger) -> None:
    """Remove and close every existing root handler before rebinding logging."""

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def _install_root_file_handler(*, role: str, verbose: bool) -> Path:
    """Install one rotating file handler on the root logger for the given role."""

    log_path = _log_path_for_role(role)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    _reset_root_handlers(root_logger)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=DEFAULT_LOG_FILE_MAX_BYTES,
        backupCount=DEFAULT_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    return log_path


@trace_function
def archive_existing_logs() -> tuple[Path | None, int]:
    """Archive loose log files into one zip and remove the originals."""

    log_files = _iter_log_artifacts(include_archives=False)
    if not log_files:
        return None, 0

    archive_path = _archive_path_for_current_session()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, mode="w", compression=_archive_compression()) as archive_file:
            for log_file in log_files:
                archive_file.write(log_file, arcname=log_file.name)
        for log_file in log_files:
            log_file.unlink()
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    return archive_path, len(log_files)


@trace_function
def cleanse_existing_logs() -> int:
    """Delete every existing log artifact from the project log directory."""

    artifacts = _iter_log_artifacts(include_archives=True)
    for artifact in artifacts:
        artifact.unlink()
    return len(artifacts)


def _configure_logger_tree(*, role: str, verbose: bool) -> logging.Logger:
    """Bind the root/app logger tree to one role-specific rotating file."""

    global _CURRENT_VERBOSE, _CURRENT_ROLE

    log_path = _install_root_file_handler(role=role, verbose=verbose)
    _CURRENT_VERBOSE = verbose
    _CURRENT_ROLE = role

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.info("Logging configured for role=%s file=%s", role, log_path.name)
    return logger


def is_verbose() -> bool:
    """Return whether the current logging configuration is in verbose mode.

    Use this when a callsite wants to gate extra operator-visible output on the
    ``--verbose`` flag without threading the flag through every function.
    """

    return _CURRENT_VERBOSE


@trace_function
def configure_logging(verbose: bool = False, *, role: str = "bootstrap") -> logging.Logger:
    """Configure file logging for the current process and return the app logger."""

    return _configure_logger_tree(role=role, verbose=verbose)


@trace_function
def rebind_logging_role(role: str, *, verbose: bool | None = None) -> logging.Logger:
    """Switch the active log file to a new role while keeping the current session timestamp."""

    effective_verbose = _CURRENT_VERBOSE if verbose is None else verbose
    if role == _CURRENT_ROLE and effective_verbose == _CURRENT_VERBOSE and logging.getLogger().handlers:
        return logging.getLogger(LOGGER_NAME)
    return _configure_logger_tree(role=role, verbose=effective_verbose)
