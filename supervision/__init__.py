"""Supervisor-role code: bootstrapping a host into the cluster and watching its peer subprocess.

Sibling of ``main_node/`` and ``compute_node/``. Holds the Supervisor entry
loop, its supervisor↔peer heartbeat/diagnostics, capacity planning, and
the compute-resource policy that shapes a host's declared capacity.
"""
