"""Classify subprocess exit codes into human-readable death causes.

Use this when a child process (peer supervisor, native compute runner) exits
unexpectedly and the operator needs to know whether the cause was an OS-level
event (out-of-memory kill, segfault, deliberate signal) versus a runner bug.

Windows reports NTSTATUS-style codes via the process exit code; Posix systems
report a negative value when the child was terminated by a signal.
"""

from __future__ import annotations

import signal
import sys


# Subset of NTSTATUS values that show up as Windows process exit codes when the
# OS or a fault handler tears the process down. Names mirror the constants in
# <ntstatus.h>; the comments describe the operator-visible cause.
_WINDOWS_STATUS_CODES: dict[int, str] = {
    0xC0000005: "STATUS_ACCESS_VIOLATION (segfault / bad pointer dereference)",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION (CPU rejected an opcode)",
    0xC0000025: "STATUS_NONCONTINUABLE_EXCEPTION",
    0xC0000026: "STATUS_INVALID_DISPOSITION",
    0xC000008C: "STATUS_ARRAY_BOUNDS_EXCEEDED",
    0xC000008D: "STATUS_FLOAT_DENORMAL_OPERAND",
    0xC000008E: "STATUS_FLOAT_DIVIDE_BY_ZERO",
    0xC000008F: "STATUS_FLOAT_INEXACT_RESULT",
    0xC0000090: "STATUS_FLOAT_INVALID_OPERATION",
    0xC0000091: "STATUS_FLOAT_OVERFLOW",
    0xC0000092: "STATUS_FLOAT_STACK_CHECK",
    0xC0000093: "STATUS_FLOAT_UNDERFLOW",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000095: "STATUS_INTEGER_OVERFLOW",
    0xC0000096: "STATUS_PRIVILEGED_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW (recursion or large stack alloc)",
    0xC0000135: "STATUS_DLL_NOT_FOUND (missing dependency DLL)",
    0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    0xC000013A: "STATUS_CONTROL_C_EXIT (Ctrl+C / external terminate)",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC00000FE: "STATUS_NOT_REGISTRY_FILE",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN (/GS or fast-fail trip)",
    0xC0000417: "STATUS_INVALID_CRUNTIME_PARAMETER",
    0xC0000420: "STATUS_ASSERTION_FAILURE",
    0x40010005: "DBG_CONTROL_C (debugger-style Ctrl+C)",
}

# Common Posix signals worth naming explicitly. Anything else falls through to
# ``signal.Signals(n).name`` if available.
_POSIX_SIGNAL_NOTES: dict[int, str] = {
    signal.SIGTERM.value if hasattr(signal, "SIGTERM") else 15: "SIGTERM (graceful terminate)",
    signal.SIGINT.value if hasattr(signal, "SIGINT") else 2: "SIGINT (Ctrl+C)",
    signal.SIGKILL.value if hasattr(signal, "SIGKILL") else 9: "SIGKILL (uncatchable kill, often OOM-killer)",
    signal.SIGSEGV.value if hasattr(signal, "SIGSEGV") else 11: "SIGSEGV (segfault)",
    signal.SIGABRT.value if hasattr(signal, "SIGABRT") else 6: "SIGABRT (abort / failed assertion)",
    signal.SIGBUS.value if hasattr(signal, "SIGBUS") else 7: "SIGBUS (bad memory access)",
    signal.SIGFPE.value if hasattr(signal, "SIGFPE") else 8: "SIGFPE (floating-point exception)",
    signal.SIGILL.value if hasattr(signal, "SIGILL") else 4: "SIGILL (illegal instruction)",
    signal.SIGPIPE.value if hasattr(signal, "SIGPIPE") else 13: "SIGPIPE (write to closed pipe)",
    signal.SIGHUP.value if hasattr(signal, "SIGHUP") else 1: "SIGHUP (terminal/parent gone)",
}


def classify_exit_code(returncode: int | None) -> str:
    """Return a one-line human description of a process exit code.

    Args:
        returncode: ``Popen.returncode`` value: ``None`` if still running,
            ``0`` for normal exit, positive for normal nonzero exit (Posix +
            Windows), or negative on Posix when a signal terminated the
            child. On Windows, large unsigned values like ``0xC0000005``
            indicate NTSTATUS-style faults.

    Returns:
        A human-readable string suitable for logging next to the raw code.
    """
    if returncode is None:
        return "still running"
    if returncode == 0:
        return "exit 0 (clean)"

    if sys.platform == "win32":
        # Treat the value as unsigned 32-bit so STATUS_xxx codes (which are
        # negative when sign-extended) match the table.
        unsigned = returncode & 0xFFFFFFFF
        named = _WINDOWS_STATUS_CODES.get(unsigned)
        if named is not None:
            return f"exit 0x{unsigned:08X} ({named})"
        if unsigned >= 0x80000000:
            return f"exit 0x{unsigned:08X} (NTSTATUS-style fault, code not in table)"
        return f"exit {returncode} (nonzero exit, runner-reported failure)"

    if returncode < 0:
        sig_value = -returncode
        note = _POSIX_SIGNAL_NOTES.get(sig_value)
        if note is not None:
            return f"signal {sig_value} ({note})"
        try:
            name = signal.Signals(sig_value).name
        except (ValueError, AttributeError):
            return f"signal {sig_value} (unknown)"
        return f"signal {sig_value} ({name})"

    return f"exit {returncode} (nonzero exit, runner-reported failure)"
