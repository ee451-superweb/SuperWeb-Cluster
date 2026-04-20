"""Internal data-plane frame models for large artifact transfer.

Use this module when Python code needs typed internal representations of the
low-level request/init/chunk/end/error frames exchanged on the data plane.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class DataPlaneMessageType(enum.IntEnum):
    """Supported low-level data-plane frame types."""

    DOWNLOAD_REQUEST = 1
    INIT = 2
    CHUNK = 3
    END = 4
    ERROR = 5
    DELIVER = 6


@dataclass(slots=True)
class ArtifactDownloadRequestFrame:
    """Typed download-request frame asking the sender for one artifact id."""

    artifact_id: str


@dataclass(slots=True)
class ArtifactDeliverFrame:
    """Typed deliver frame announcing a client-initiated upload on this socket."""

    upload_id: str
    size_bytes: int
    checksum: str
    content_type: str


@dataclass(slots=True)
class ArtifactInitFrame:
    """Typed init frame announcing artifact metadata before streaming bytes."""

    size_bytes: int
    chunk_size: int
    checksum: str
    content_type: str


@dataclass(slots=True)
class ArtifactChunkFrame:
    """Typed chunk frame carrying one artifact byte range."""

    offset: int
    data: bytes


@dataclass(slots=True)
class ArtifactEndFrame:
    """Typed end frame signaling a completed artifact transfer."""

    size_bytes: int


@dataclass(slots=True)
class ArtifactErrorFrame:
    """Typed error frame terminating a data-plane transfer with a message."""

    message: str
