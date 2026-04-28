"""Worker-side conv2d weight artifact fetch tests."""

from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from core.config import AppConfig
from core.constants import COMPUTE_NODE_NAME, METHOD_CONV2D, STATUS_OK
from core.float32_codec import pack_float32_values
from compute_node.worker_services import WorkerTaskRuntimeService
from transport.artifact_manager import ArtifactManager
from wire.internal_protocol.control_plane import Conv2dTaskPayload
from wire.internal_protocol.control_plane_codec import encode_envelope, parse_envelope
from wire.internal_protocol.transport import (
    MessageKind,
    TaskAssign,
    TaskResult,
    TransferMode,
    build_task_assign,
)


class _RecordingSession:
    def __init__(self) -> None:
        self.sent_messages = []

    def send(self, message) -> None:
        self.sent_messages.append(message)


class _CapturingExecutor:
    def __init__(self) -> None:
        self.tasks: list[TaskAssign] = []
        self.started = threading.Event()

    def execute_task(self, task) -> TaskResult:
        self.tasks.append(task)
        self.started.set()
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            iteration_count=task.iteration_count,
        )


def _build_conv2d_task_assign(*, weight_artifact, weight_data: bytes = b"") -> TaskAssign:
    envelope = build_task_assign(
        request_id="req-conv2d",
        node_id=COMPUTE_NODE_NAME,
        task_id="req-conv2d:worker-1:0",
        method=METHOD_CONV2D,
        size="small",
        object_id="conv2d/small",
        stream_id="stream-conv2d",
        iteration_count=1,
        transfer_mode=TransferMode.ARTIFACT_REQUIRED,
        artifact_id="req-conv2d:worker-1:0",
        task_payload=Conv2dTaskPayload(
            start_oc=0,
            end_oc=2,
            tensor_h=8,
            tensor_w=8,
            channels_in=4,
            channels_out=2,
            kernel_size=3,
            padding=1,
            stride=1,
            weight_data=weight_data,
            weight_artifact=weight_artifact,
        ),
    )
    return envelope.task_assign


class WorkerWeightArtifactFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.artifact_manager = ArtifactManager(
            root_dir=Path(self._tempdir.name) / "artifacts",
            public_host="127.0.0.1",
            chunk_size=16 * 1024,
        )
        self.addCleanup(self.artifact_manager.close)
        self.service = WorkerTaskRuntimeService(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            logger=mock.Mock(),
            node_name=COMPUTE_NODE_NAME,
        )

    def test_worker_fetches_weight_artifact_into_payload(self) -> None:
        weight_bytes = pack_float32_values([0.25] * (3 * 3 * 4 * 2))
        descriptor = self.artifact_manager.publish_bytes(
            weight_bytes,
            producer_node_id="main-node",
            artifact_id="main-slice-weight-1",
            content_type="application/x-superweb-conv2d-weight",
        )
        task = _build_conv2d_task_assign(weight_artifact=descriptor)
        session = _RecordingSession()
        executor = _CapturingExecutor()
        pending_tasks: dict[str, tuple[object, object]] = {}
        with ThreadPoolExecutor(max_workers=1) as thread_pool:
            self.service.handle_task_assign_message(
                session=session,
                assigned_node_id="worker-1",
                task=task,
                pending_tasks=pending_tasks,
                process_pool=None,
                thread_pool=thread_pool,
                task_executor=executor,
                subprocess_entrypoint=None,
                artifact_manager=self.artifact_manager,
            )
            self.assertTrue(executor.started.wait(timeout=5.0))

        self.assertEqual(len(executor.tasks), 1)
        submitted_task = executor.tasks[0]
        self.assertIsInstance(submitted_task.task_payload, Conv2dTaskPayload)
        self.assertEqual(submitted_task.task_payload.weight_data, weight_bytes)
        self.assertGreaterEqual(len(session.sent_messages), 1)
        self.assertEqual(session.sent_messages[0].kind, MessageKind.TASK_ACCEPT)

    def test_worker_fails_task_when_weight_artifact_fetch_fails(self) -> None:
        weight_bytes = b"ignored"
        descriptor = self.artifact_manager.publish_bytes(
            weight_bytes,
            producer_node_id="main-node",
            artifact_id="main-slice-weight-missing",
        )
        self.artifact_manager.remove_artifact(descriptor.artifact_id)

        task = _build_conv2d_task_assign(weight_artifact=descriptor)
        session = _RecordingSession()
        executor = _CapturingExecutor()
        pending_tasks: dict[str, tuple[object, object]] = {}
        with ThreadPoolExecutor(max_workers=1) as thread_pool:
            self.service.handle_task_assign_message(
                session=session,
                assigned_node_id="worker-1",
                task=task,
                pending_tasks=pending_tasks,
                process_pool=None,
                thread_pool=thread_pool,
                task_executor=executor,
                subprocess_entrypoint=None,
                artifact_manager=self.artifact_manager,
            )

        self.assertEqual(executor.tasks, [])
        self.assertEqual(pending_tasks, {})
        self.assertEqual(len(session.sent_messages), 2)
        self.assertEqual(session.sent_messages[0].kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(session.sent_messages[1].kind, MessageKind.TASK_FAIL)
        self.assertEqual(session.sent_messages[1].task_fail.task_id, task.task_id)
        self.assertIn("weight", session.sent_messages[1].task_fail.error_message)

    def test_worker_fails_task_when_artifact_manager_is_missing(self) -> None:
        from wire.internal_protocol.control_plane import ArtifactDescriptor

        descriptor = ArtifactDescriptor(
            artifact_id="needs-manager",
            content_type="application/x-superweb-conv2d-weight",
            size_bytes=16,
            checksum="deadbeef",
            producer_node_id="main-node",
            transfer_host="10.0.0.5",
            transfer_port=52030,
            chunk_size=65536,
            ready=True,
        )
        task = _build_conv2d_task_assign(weight_artifact=descriptor)
        session = _RecordingSession()
        executor = _CapturingExecutor()
        pending_tasks: dict[str, tuple[object, object]] = {}

        with ThreadPoolExecutor(max_workers=1) as thread_pool:
            self.service.handle_task_assign_message(
                session=session,
                assigned_node_id="worker-1",
                task=task,
                pending_tasks=pending_tasks,
                process_pool=None,
                thread_pool=thread_pool,
                task_executor=executor,
                subprocess_entrypoint=None,
                artifact_manager=None,
            )

        self.assertEqual(executor.tasks, [])
        self.assertEqual(pending_tasks, {})
        self.assertEqual(len(session.sent_messages), 2)
        self.assertEqual(session.sent_messages[0].kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(session.sent_messages[1].kind, MessageKind.TASK_FAIL)
        self.assertEqual(session.sent_messages[1].task_fail.task_id, task.task_id)
        self.assertIn(
            "artifact manager is required",
            session.sent_messages[1].task_fail.error_message,
        )

    def test_oversized_weight_slice_survives_artifact_transfer(self) -> None:
        large_payload = b"\xA5" * (8 * 1024 * 1024)
        self.assertGreater(len(large_payload), AppConfig().max_message_size)
        descriptor = self.artifact_manager.publish_bytes(
            large_payload,
            producer_node_id="main-node",
            artifact_id="main-slice-weight-large",
        )
        task = _build_conv2d_task_assign(weight_artifact=descriptor)
        session = _RecordingSession()
        executor = _CapturingExecutor()
        pending_tasks: dict[str, tuple[object, object]] = {}
        with ThreadPoolExecutor(max_workers=1) as thread_pool:
            self.service.handle_task_assign_message(
                session=session,
                assigned_node_id="worker-1",
                task=task,
                pending_tasks=pending_tasks,
                process_pool=None,
                thread_pool=thread_pool,
                task_executor=executor,
                subprocess_entrypoint=None,
                artifact_manager=self.artifact_manager,
            )
            self.assertTrue(executor.started.wait(timeout=10.0))

        self.assertEqual(len(executor.tasks), 1)
        self.assertEqual(executor.tasks[0].task_payload.weight_data, large_payload)


class WeightArtifactCodecRoundTripTests(unittest.TestCase):
    def test_conv2d_task_payload_weight_artifact_survives_codec(self) -> None:
        descriptor = self._make_descriptor()
        envelope = build_task_assign(
            request_id="req-codec",
            node_id=COMPUTE_NODE_NAME,
            task_id="req-codec:worker-1:0",
            method=METHOD_CONV2D,
            size="small",
            object_id="conv2d/small",
            stream_id="stream-codec",
            iteration_count=1,
            transfer_mode=TransferMode.ARTIFACT_REQUIRED,
            artifact_id="req-codec:worker-1:0",
            task_payload=Conv2dTaskPayload(
                start_oc=0,
                end_oc=2,
                tensor_h=8,
                tensor_w=8,
                channels_in=4,
                channels_out=2,
                kernel_size=3,
                padding=1,
                stride=1,
                weight_data=b"",
                weight_artifact=descriptor,
            ),
        )

        round_tripped = parse_envelope(encode_envelope(envelope))

        payload = round_tripped.task_assign.task_payload
        self.assertIsInstance(payload, Conv2dTaskPayload)
        self.assertIsNotNone(payload.weight_artifact)
        self.assertEqual(payload.weight_artifact.artifact_id, descriptor.artifact_id)
        self.assertEqual(payload.weight_artifact.transfer_host, descriptor.transfer_host)
        self.assertEqual(payload.weight_artifact.transfer_port, descriptor.transfer_port)
        self.assertEqual(payload.weight_artifact.size_bytes, descriptor.size_bytes)
        self.assertEqual(payload.weight_artifact.checksum, descriptor.checksum)
        self.assertEqual(payload.weight_data, b"")

    @staticmethod
    def _make_descriptor():
        from wire.internal_protocol.control_plane import ArtifactDescriptor

        return ArtifactDescriptor(
            artifact_id="slice-1",
            content_type="application/x-superweb-conv2d-weight",
            size_bytes=4096,
            checksum="deadbeef",
            producer_node_id="main-node",
            transfer_host="10.0.0.5",
            transfer_port=52030,
            chunk_size=65536,
            ready=True,
        )


if __name__ == "__main__":
    unittest.main()
