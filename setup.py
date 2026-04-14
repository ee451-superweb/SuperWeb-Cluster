"""Project environment setup helpers for superweb-cluster.

This file intentionally separates:

- local-only setup work such as creating `.venv`
- potentially networked setup work such as `pip install -r requirements.txt`

`bootstrap.py` imports these helpers so the startup path stays small and
humans can tell which steps may need internet access.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP_PATH = VENV_DIR / ".requirements.sha256"


@dataclass(slots=True)
class ProjectEnvironmentStatus:
    """Observed state of the local project Python environment."""

    venv_exists: bool
    requirements_current: bool
    using_project_python: bool

    @property
    def ready(self) -> bool:
        """Return whether bootstrap can safely rely on the project environment."""

        return self.venv_exists and self.requirements_current


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone setup CLI."""

    parser = argparse.ArgumentParser(
        description="Prepare the superweb-cluster local Python environment.",
    )
    parser.add_argument(
        "--venv-only",
        action="store_true",
        help="Create or refresh the local .venv only. This is a local-only step and does not install dependencies.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed setup logging.",
    )
    return parser


def configure_logger(verbose: bool = False) -> logging.Logger:
    """Create a simple setup logger without relying on project runtime modules."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return logging.getLogger("superweb-cluster.setup")


def display_project_path(path: Path) -> str:
    """Render a project-local path for user-facing log messages."""

    return path.relative_to(PROJECT_ROOT).as_posix()


def project_python_path() -> Path:
    """Return the Python executable inside the project's virtual environment."""

    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def active_python_path() -> Path:
    """Prefer the project venv interpreter when it already exists."""

    venv_python = project_python_path()
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def current_python_uses_project_venv() -> bool:
    """Return whether the current interpreter is the project's venv Python."""

    venv_python = project_python_path()
    if not venv_python.exists():
        return False
    try:
        return Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        return Path(sys.executable) == venv_python


def requirements_hash() -> str:
    """Fingerprint `requirements.txt` so installs only rerun when it changes."""

    if not REQUIREMENTS_PATH.exists():
        return ""
    return hashlib.sha256(REQUIREMENTS_PATH.read_bytes()).hexdigest()


def requirements_are_current() -> bool:
    """Return whether the installed dependency stamp matches requirements.txt."""

    wanted_hash = requirements_hash()
    if not wanted_hash:
        return True
    if not REQUIREMENTS_STAMP_PATH.exists():
        return False
    installed_hash = REQUIREMENTS_STAMP_PATH.read_text(encoding="utf-8").strip()
    return wanted_hash == installed_hash


def inspect_project_environment() -> ProjectEnvironmentStatus:
    """Inspect whether `.venv` exists, dependencies are current, and the current interpreter matches it."""

    return ProjectEnvironmentStatus(
        venv_exists=project_python_path().exists(),
        requirements_current=requirements_are_current(),
        using_project_python=current_python_uses_project_venv(),
    )


def ensure_virtual_environment(logger: logging.Logger) -> bool:
    """Create `.venv` when missing.

    This step is local-only and should not require network access.
    """

    venv_python = project_python_path()
    if venv_python.exists():
        logger.info("Local Python environment already exists at %s.", display_project_path(VENV_DIR))
        return True

    logger.info("Creating local virtual environment at %s.", display_project_path(VENV_DIR))
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Failed to create the local virtual environment: %s", exc)
        return False
    return True


def install_project_requirements(logger: logging.Logger) -> bool:
    """Install `requirements.txt` when the hash changed.

    This step may require network access because it runs `pip install`.
    """

    wanted_hash = requirements_hash()
    if not wanted_hash:
        logger.info("No requirements.txt found, skipping dependency installation.")
        return True

    installed_hash = REQUIREMENTS_STAMP_PATH.read_text(encoding="utf-8").strip() if REQUIREMENTS_STAMP_PATH.exists() else ""
    if wanted_hash == installed_hash:
        logger.info("Requirements are already up to date according to %s.", display_project_path(REQUIREMENTS_STAMP_PATH))
        return True

    logger.info(
        "Installing dependencies from %s using %s. This step may require network access.",
        display_project_path(REQUIREMENTS_PATH),
        display_project_path(project_python_path()),
    )
    try:
        subprocess.run(
            [str(project_python_path()), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Failed to install project dependencies: %s", exc)
        return False

    REQUIREMENTS_STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    REQUIREMENTS_STAMP_PATH.write_text(wanted_hash, encoding="utf-8")
    return True


def ensure_project_python_environment(logger: logging.Logger, *, install_requirements_flag: bool = True) -> bool:
    """Ensure the local Python environment is ready for the project."""

    if not ensure_virtual_environment(logger):
        return False
    if not install_requirements_flag:
        logger.info("Skipping dependency installation because --venv-only was requested.")
        return True
    return install_project_requirements(logger)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone setup entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logger(verbose=args.verbose)
    ready = ensure_project_python_environment(
        logger,
        install_requirements_flag=not args.venv_only,
    )
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
