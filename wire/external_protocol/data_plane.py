"""External data-plane models shared across nodes and clients.

Use this module when the control plane needs to describe externally fetchable
artifact content without inlining its bytes into runtime envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ArtifactDescriptor:
    """Describe one published artifact that can be fetched over the data plane."""

    artifact_id: str
    content_type: str
    size_bytes: int
    checksum: str
    producer_node_id: str
    transfer_host: str
    transfer_port: int
    chunk_size: int
    ready: bool = True
