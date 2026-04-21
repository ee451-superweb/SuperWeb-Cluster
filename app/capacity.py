"""Read the benchmark result file to learn which backends this host can run.

Use this module when the supervisor needs to know which compute backends are
usable on the current machine without re-probing them. The benchmark already
recorded that information during the last ``result.json`` generation; this
loader just unions the per-method ``usable_backends`` lists into one set.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_usable_backends(result_path: Path) -> set[str]:
    """Return the union of usable backends across every method in ``result.json``.

    Use this when deciding whether ``--backend`` / ``--dual-purpose`` requests
    can be honored, or when picking a default backend for the current host.

    Args:
        result_path: Absolute path to the benchmark ``result.json``.

    Returns:
        The set of backend names that at least one method reported as usable.
        Returns an empty set when the file is missing or malformed.
    """
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()

    methods = payload.get("methods")
    if not isinstance(methods, dict):
        return set()

    usable: set[str] = set()
    for method_report in methods.values():
        if not isinstance(method_report, dict):
            continue
        for backend_name in method_report.get("usable_backends") or ():
            if isinstance(backend_name, str) and backend_name:
                usable.add(backend_name)
    return usable
