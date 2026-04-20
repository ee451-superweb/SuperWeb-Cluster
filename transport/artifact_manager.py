"""Publish, retain, reap, and fetch artifacts exposed over the data plane.

Use this module when runtime code wants a higher-level facade over the raw
large-data transport: publish bytes or files, register an existing file, fetch
an artifact by descriptor, and clean up temporary local storage automatically.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from transport.large_data_transfer import (
    LargeDataTransferServer,
    PublishedArtifact,
    UploadSlot,
    fetch_artifact_to_file,
    publish_file_descriptor,
)
from wire.external_protocol.data_plane import ArtifactDescriptor


@dataclass(slots=True)
class ManagedArtifact:
    """Track one published artifact plus its optional expiry and cleanup policy."""

    published: PublishedArtifact
    expires_at_unix_ms: int | None = None
    delete_local_path: bool = True


class ArtifactManager:
    """Own local artifacts and expose them over the data plane."""

    def __init__(
        self,
        *,
        root_dir: Path,
        public_host: str,
        port: int = 0,
        chunk_size: int = 512 * 1024,
    ) -> None:
        """Use this when one process needs to own artifact storage and serving state.

        Args: root_dir local artifact directory, public_host externally reachable host, port optional listen port, chunk_size preferred stream chunk size.
        Returns: None after local storage and the embedded data-plane server are prepared.
        """
        self.root_dir = Path(root_dir)
        self.public_host = public_host
        self.chunk_size = chunk_size
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, ManagedArtifact] = {}
        self._upload_slots: dict[str, UploadSlot] = {}
        self._upload_lock = threading.Lock()
        self._server = LargeDataTransferServer(
            host="",
            port=port,
            resolve_artifact=self._resolve_artifact,
            resolve_upload_slot=self._resolve_upload_slot,
            consume_upload_slot=self._consume_upload_slot,
            chunk_size=chunk_size,
        )
        self._reaper_stop_event = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Use this after server startup when callers need the exposed TCP port.

        Args: self artifact manager whose embedded server port is queried.
        Returns: The current data-plane TCP port.
        """
        return self._server.port

    def start(self) -> None:
        """Use this before publishing or serving artifacts to ensure the server is running.

        Args: self artifact manager that should expose its data-plane endpoint.
        Returns: None after the data-plane server and reaper thread are running.
        """
        self._server.start()
        if self._reaper_thread is None or not self._reaper_thread.is_alive():
            self._reaper_stop_event.clear()
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop,
                name="artifact-reaper",
                daemon=True,
            )
            self._reaper_thread.start()

    def close(self) -> None:
        """Use this during shutdown to stop serving and delete managed artifacts.

        Args: self artifact manager whose server and local storage should be closed.
        Returns: None after the server stops and managed artifacts are removed.
        """
        self._reaper_stop_event.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=1.0)
            self._reaper_thread = None
        self._server.close()
        for artifact_id in list(self._artifacts):
            self.remove_artifact(artifact_id)

    def set_public_host(self, host: str) -> None:
        """Use this when the externally advertised artifact host should change.

        Args: host new hostname or IP to embed into future artifact descriptors.
        Returns: None after the manager stores the new advertised host.
        """
        self.public_host = host

    def _resolve_artifact(self, artifact_id: str) -> PublishedArtifact | None:
        """Use this internal callback when the data-plane server resolves an artifact id.

        Args: artifact_id protocol-level artifact identifier requested by a client.
        Returns: The published artifact record, or None when it is missing or expired.
        """
        self._prune_expired_artifacts()
        managed = self._artifacts.get(artifact_id)
        return None if managed is None else managed.published

    def _resolve_upload_slot(self, upload_id: str) -> UploadSlot | None:
        """Use this internal callback when the data-plane server validates an incoming DELIVER.

        Args: upload_id identifier the client attached to its upload attempt.
        Returns: The matching pre-registered upload slot, or None when unknown.
        """
        with self._upload_lock:
            return self._upload_slots.get(upload_id)

    def _consume_upload_slot(self, upload_id: str) -> None:
        """Use this internal callback after an upload slot has been served or failed.

        Args: upload_id identifier of the slot whose lifecycle has completed.
        Returns: None after the slot is removed from the registry.
        """
        with self._upload_lock:
            self._upload_slots.pop(upload_id, None)

    def register_upload_slot(
        self,
        *,
        upload_id: str,
        expected_size: int,
        expected_checksum: str = "",
        expected_content_type: str = "",
        destination_suffix: str = ".upload.bin",
    ) -> concurrent.futures.Future:
        """Use this when the control plane has negotiated an upload and needs a landing slot.

        Args: upload_id main-assigned id that the client will carry in its DELIVER frame, expected_size declared byte count to enforce, expected_checksum optional SHA-256 to validate at END, expected_content_type optional content label to cross-check, destination_suffix local filename suffix used on disk.
        Returns: A Future that resolves to the final local path once the upload completes, or raises if the upload is rejected.
        """
        self.start()
        future: concurrent.futures.Future = concurrent.futures.Future()
        destination_path = self._storage_path_for_artifact_id(upload_id, suffix=destination_suffix)
        slot = UploadSlot(
            upload_id=upload_id,
            expected_size=expected_size,
            expected_checksum=expected_checksum,
            expected_content_type=expected_content_type,
            destination_path=destination_path,
            completion_future=future,
        )
        with self._upload_lock:
            if upload_id in self._upload_slots:
                raise ValueError(f"upload slot already registered: {upload_id}")
            self._upload_slots[upload_id] = slot
        return future

    def cancel_upload_slot(self, upload_id: str) -> None:
        """Use this to retire an upload slot that will never be consumed.

        Args: upload_id identifier of the slot to remove.
        Returns: None. If the slot is still pending its future is cancelled.
        """
        with self._upload_lock:
            slot = self._upload_slots.pop(upload_id, None)
        if slot is not None and not slot.completion_future.done():
            slot.completion_future.cancel()

    def _reaper_loop(self) -> None:
        """Use this background loop to periodically delete expired managed artifacts.

        Args: self artifact manager whose managed artifacts should be reaped.
        Returns: None after the stop event ends the reaper thread.
        """
        while not self._reaper_stop_event.wait(5.0):
            self._prune_expired_artifacts()

    def _prune_expired_artifacts(self) -> None:
        """Use this internal helper whenever expiry checks should be applied eagerly.

        Args: self artifact manager whose expiry table should be scanned.
        Returns: None after expired artifacts are removed from local storage and registry.
        """
        now_ms = int(time.time() * 1000)
        for artifact_id, managed in list(self._artifacts.items()):
            if managed.expires_at_unix_ms is not None and managed.expires_at_unix_ms <= now_ms:
                self.remove_artifact(artifact_id)

    def _register_published_artifact(
        self,
        artifact_id: str,
        published: PublishedArtifact,
        *,
        ttl_seconds: float | None,
        delete_local_path: bool,
    ) -> ArtifactDescriptor:
        """Use this internal helper after turning a local file into a published artifact.

        Args: artifact_id protocol id, published descriptor/path pair, ttl_seconds optional retention TTL, delete_local_path cleanup policy.
        Returns: The control-plane descriptor that callers should send to peers.
        """
        expires_at_unix_ms = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at_unix_ms = int(time.time() * 1000 + ttl_seconds * 1000)
        self._artifacts[artifact_id] = ManagedArtifact(
            published=published,
            expires_at_unix_ms=expires_at_unix_ms,
            delete_local_path=delete_local_path,
        )
        return published.descriptor

    def remove_artifact(self, artifact_id: str) -> bool:
        """Use this when an artifact should be removed explicitly or after expiry.

        Args: artifact_id protocol-level artifact identifier to delete.
        Returns: True when an artifact was removed, otherwise False.
        """
        managed = self._artifacts.pop(artifact_id, None)
        if managed is None:
            return False
        if managed.delete_local_path:
            managed.published.local_path.unlink(missing_ok=True)
        return True

    def _storage_path_for_artifact_id(self, artifact_id: str, *, suffix: str) -> Path:
        """Use this internal helper to map an artifact id onto a safe local filename.

        Args: artifact_id protocol identifier and suffix desired local filename suffix.
        Returns: A deterministic local path under the manager root directory.
        """

        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact_id).strip("._")
        if not normalized:
            normalized = "artifact"
        digest = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:12]
        return self.root_dir / f"{normalized[:80]}-{digest}{suffix}"

    def publish_bytes(
        self,
        payload: bytes,
        *,
        producer_node_id: str,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> ArtifactDescriptor:
        """Use this when result bytes already exist in memory and should become an artifact.

        Args: payload bytes to persist, producer_node_id source node id, content_type optional content label, artifact_id optional explicit id, ttl_seconds optional retention TTL.
        Returns: An artifact descriptor ready for control-plane transport.
        """
        self.start()
        artifact_id = artifact_id or uuid.uuid4().hex
        local_path = self._storage_path_for_artifact_id(artifact_id, suffix=".bin")
        local_path.write_bytes(payload)
        published = publish_file_descriptor(
            artifact_id=artifact_id,
            local_path=local_path,
            public_host=self.public_host,
            public_port=self.port,
            producer_node_id=producer_node_id,
            chunk_size=self.chunk_size,
            content_type=content_type,
        )
        return self._register_published_artifact(
            artifact_id,
            published,
            ttl_seconds=ttl_seconds,
            delete_local_path=True,
        )

    def publish_file(
        self,
        source_path: Path,
        *,
        producer_node_id: str,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
        move: bool = False,
        ttl_seconds: float | None = None,
    ) -> ArtifactDescriptor:
        """Use this when one existing file should be copied or moved into managed storage.

        Args: source_path original file, producer_node_id source node id, content_type optional label, artifact_id optional explicit id, move whether to move instead of copy, ttl_seconds optional retention TTL.
        Returns: An artifact descriptor for the published managed copy.
        """
        self.start()
        artifact_id = artifact_id or uuid.uuid4().hex
        destination_path = self._storage_path_for_artifact_id(
            artifact_id,
            suffix=source_path.suffix or ".bin",
        )
        if source_path.resolve() == destination_path.resolve():
            pass
        elif move:
            os.replace(source_path, destination_path)
        else:
            shutil.copyfile(source_path, destination_path)
        published = publish_file_descriptor(
            artifact_id=artifact_id,
            local_path=destination_path,
            public_host=self.public_host,
            public_port=self.port,
            producer_node_id=producer_node_id,
            chunk_size=self.chunk_size,
            content_type=content_type,
        )
        return self._register_published_artifact(
            artifact_id,
            published,
            ttl_seconds=ttl_seconds,
            delete_local_path=True,
        )

    def register_existing_file(
        self,
        local_path: Path,
        *,
        producer_node_id: str,
        content_type: str = "application/octet-stream",
        artifact_id: str | None = None,
        ttl_seconds: float | None = None,
        delete_local_path: bool = True,
    ) -> ArtifactDescriptor:
        """Use this when a file already lives in the right place and should just be exposed.

        Args: local_path file to expose, producer_node_id source node id, content_type optional label, artifact_id optional explicit id, ttl_seconds optional TTL, delete_local_path cleanup policy.
        Returns: An artifact descriptor pointing at the registered local file.
        """
        self.start()
        artifact_id = artifact_id or uuid.uuid4().hex
        published = publish_file_descriptor(
            artifact_id=artifact_id,
            local_path=local_path,
            public_host=self.public_host,
            public_port=self.port,
            producer_node_id=producer_node_id,
            chunk_size=self.chunk_size,
            content_type=content_type,
        )
        return self._register_published_artifact(
            artifact_id,
            published,
            ttl_seconds=ttl_seconds,
            delete_local_path=delete_local_path,
        )

    def fetch_to_file(self, descriptor: ArtifactDescriptor, destination_path: Path, *, timeout: float) -> Path:
        """Use this when a remote artifact should be downloaded straight to disk.

        Args: descriptor remote artifact descriptor, destination_path local output path, timeout socket timeout in seconds.
        Returns: The destination path after a successful download and validation.
        """
        return fetch_artifact_to_file(descriptor, destination_path, timeout=timeout)

    def fetch_bytes(self, descriptor: ArtifactDescriptor, *, timeout: float) -> bytes:
        """Use this when callers need a remote artifact materialized back into memory.

        Args: descriptor remote artifact descriptor and timeout socket timeout in seconds.
        Returns: The fetched artifact bytes after download and validation.
        """
        tmp_path = self._storage_path_for_artifact_id(f"fetch-{descriptor.artifact_id}", suffix=".bin")
        try:
            fetch_artifact_to_file(descriptor, tmp_path, timeout=timeout)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
