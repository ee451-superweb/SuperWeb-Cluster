"""Project environment setup helpers for superweb-cluster.

This file intentionally separates:

- local-only setup work such as creating `.venv`
- potentially networked setup work such as `pip install -r requirements.txt`

`bootstrap.py` imports these helpers so the startup path stays small and
humans can tell which steps may need internet access.
"""

from __future__ import annotations

import sys

MIN_PYTHON = (3, 11)
DEV_PYTHON_DISPLAY = "3.14.3"
if sys.version_info < MIN_PYTHON:
    _found = ".".join(str(n) for n in sys.version_info[:3])
    _required = ".".join(str(n) for n in MIN_PYTHON)
    sys.stderr.write(
        f"superweb-cluster setup.py requires Python {_required}+, "
        f"but this interpreter is Python {_found}.\n"
        f"The project is developed on Python {DEV_PYTHON_DISPLAY}. Install a supported Python via one of:\n"
        f"  - uv python install 3.14   (https://docs.astral.sh/uv/)\n"
        f"  - https://www.python.org/downloads/\n"
        f"  - winget install Python.Python.3.14   (Windows)\n"
        f"  - brew install python@3.14            (macOS)\n"
    )
    sys.exit(1)

import argparse
import hashlib
import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from adapters.process import enable_utf8_mode, python_utf8_command

enable_utf8_mode()

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP_PATH = VENV_DIR / ".requirements.sha256"

def _probe_msvc_via_vswhere(_tool: dict) -> tuple[bool, str | None, str]:
    """Detect MSVC via vswhere; `cl.exe` is never on the default PATH by design."""

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return False, None, "vswhere.exe not found (no Visual Studio Installer present)"
    try:
        result = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, None, "vswhere.exe failed to run"
    path = (result.stdout or "").strip()
    if not path:
        return False, None, "Visual Studio present but no C++ workload installed"
    return True, f"installed at {path}", ""


DEV_TOOLS: tuple[dict, ...] = (
    {
        "name": "MSVC (cl.exe)",
        "probe_impl": _probe_msvc_via_vswhere,
        "applicable_on": frozenset({"Windows"}),
        "install": {
            "Windows": "winget install Microsoft.VisualStudio.2022.BuildTools",
        },
        "landing": "https://visualstudio.microsoft.com/downloads/",
    },
    {
        "name": "NVIDIA CUDA (nvcc)",
        "probe_cmd": ["nvcc", "--version"],
        "applicable_on": frozenset({"Windows", "Linux"}),
        "install": {
            "Windows": "winget install Nvidia.CUDA",
            "Linux": "sudo apt install nvidia-cuda-toolkit",
        },
        "landing": "https://developer.nvidia.com/cuda-downloads",
    },
    {
        "name": ".NET SDK (>= 8.0; latest LTS is 10)",
        "probe_cmd": ["dotnet", "--list-sdks"],
        "min_version": (8, 0),
        "applicable_on": frozenset({"Windows", "Linux", "Darwin"}),
        "install": {
            "Windows": "winget install Microsoft.DotNet.SDK.10   (latest LTS)",
            "Linux": "sudo apt install dotnet-sdk-10.0           (latest LTS)",
            "Darwin": "brew install --cask dotnet-sdk            (latest LTS)",
        },
        "landing": "https://dotnet.microsoft.com/download/dotnet/10.0",
    },
    {
        "name": "Xcode Command Line Tools",
        "probe_cmd": ["xcrun", "--version"],
        "applicable_on": frozenset({"Darwin"}),
        "install": {
            "Darwin": "xcode-select --install",
        },
        "landing": "https://developer.apple.com/xcode/",
    },
    {
        "name": "GCC (g++)",
        "probe_cmd": ["gcc", "--version"],
        "applicable_on": frozenset({"Windows", "Linux", "Darwin"}),
        "install": {
            "Windows": "winget install GnuWin32.Gcc   (or install MSYS2 for a full toolchain)",
            "Linux": "sudo apt install build-essential",
            "Darwin": "xcode-select --install   (ships clang, which most GCC-targeted builds accept)",
        },
        "landing": "https://gcc.gnu.org/",
    },
)


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
        "--dev",
        action="store_true",
        help=(
            "After the normal setup, print a developer-toolchain report: "
            "detect MSVC, CUDA, .NET 8, Xcode CLT, and GCC, and for any missing "
            "item show the package-manager command plus the official landing URL. "
            "Informational only; setup.py never installs OS toolchains itself."
        ),
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

    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except OSError:
        return Path(sys.prefix) == VENV_DIR


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
            python_utf8_command(sys.executable, "-m", "venv", VENV_DIR),
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
            python_utf8_command(project_python_path(), "-m", "pip", "install", "-r", REQUIREMENTS_PATH),
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


def _probe_dev_tool(tool: dict) -> tuple[bool, str | None, str]:
    """Probe one tool. Returns (satisfied, version_line, miss_reason).

    `miss_reason` distinguishes a missing binary from a binary that is
    present but does not satisfy the requested version filter.
    """

    custom = tool.get("probe_impl")
    if custom is not None:
        return custom(tool)

    exe = tool["probe_cmd"][0]
    if not shutil.which(exe):
        return False, None, "not found on PATH"
    try:
        result = subprocess.run(
            tool["probe_cmd"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True, None, ""
    combined = (result.stdout or "") + (result.stderr or "")
    all_lines = [line.strip() for line in combined.splitlines() if line.strip()]
    min_version = tool.get("min_version")
    if min_version:
        leading_version_re = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
        satisfying: list[tuple[tuple[int, ...], str]] = []
        for line in all_lines:
            match = leading_version_re.match(line)
            if not match:
                continue
            parsed = tuple(int(g) for g in match.groups() if g is not None)
            if parsed >= min_version:
                satisfying.append((parsed, line))
        if not satisfying:
            installed = ", ".join(all_lines) if all_lines else "none"
            want = ".".join(str(n) for n in min_version)
            return False, None, f"on PATH but no version >= {want} (installed: {installed})"
        satisfying.sort(key=lambda entry: entry[0])
        return True, satisfying[-1][1], ""
    first_line = all_lines[0] if all_lines else None
    return True, first_line, ""


def report_dev_toolchain(logger: logging.Logger) -> None:
    """Print a dev-toolchain detection report for the current OS.

    Does not install anything. Missing tools get a package-manager hint
    plus the official landing URL so operators can grab the latest
    release themselves.
    """

    current_os = platform.system()
    logger.info("=== Dev Toolchain Report (%s) ===", current_os or "unknown")
    for tool in DEV_TOOLS:
        name = tool["name"]
        if current_os not in tool["applicable_on"]:
            applicable = ", ".join(sorted(tool["applicable_on"]))
            logger.info("[--] %s: only applicable on %s (skipped on %s)", name, applicable, current_os)
            continue
        present, version, miss_reason = _probe_dev_tool(tool)
        if present:
            logger.info("[ok] %s: %s", name, version or "detected (version unknown)")
            continue
        logger.info("[--] %s: %s", name, miss_reason)
        install_cmd = tool["install"].get(current_os)
        if install_cmd:
            logger.info("     install: %s", install_cmd)
        logger.info("     manual:  %s", tool["landing"])
    logger.info("Note: this is informational only. setup.py never installs OS toolchains.")


def main(argv: list[str] | None = None) -> int:
    """Run the standalone setup entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logger(verbose=args.verbose)
    ready = ensure_project_python_environment(
        logger,
        install_requirements_flag=not args.venv_only,
    )
    if args.dev:
        report_dev_toolchain(logger)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
