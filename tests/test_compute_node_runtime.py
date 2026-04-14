"""Compute-node runtime tests."""

import unittest
from unittest import mock

from common.float32_codec import pack_float32_values
from common.types import DiscoveryResult, HardwareProfile
from compute_node.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile
from compute_node.runtime import ComputeNodeRuntime
from app.config import AppConfig
from app.constants import (
    COMPUTE_NODE_NAME,
    MAIN_NODE_NAME,
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
    STATUS_OK,
)
from wire.runtime import Heartbeat, MessageKind, RegisterOk, RuntimeEnvelope, TaskAssign, TaskResult


class _FakeSession:
    def __init__(self, messages: list[RuntimeEnvelope | None], register_ok: RegisterOk) -> None:
        self.messages = list(messages)
        self.register_ok = register_ok
        self.connected = False
        self.closed = False
        self.register_args = None
        self.sent_messages = []

    def connect(self) -> None:
        self.connected = True

    def register(self, node_name, hardware, performance):
        self.register_args = (node_name, hardware, performance)
        return self.register_ok

    def receive(self):
        return self.messages.pop(0)

    def send(self, message) -> None:
        self.sent_messages.append(message)

    def close(self) -> None:
        self.closed = True


class _FakeExecutor:
    def __init__(self, inventory: RuntimeProcessorInventory) -> None:
        self.inventory = inventory
        self.tasks = []

    def execute_task(self, task) -> TaskResult:
        self.tasks.append(task)
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            row_start=task.row_start,
            row_end=task.row_end,
            output_length=task.row_end - task.row_start,
            output_vector=pack_float32_values([1.0] * (task.row_end - task.row_start)),
            iteration_count=task.iteration_count,
        )


class ComputeNodeRuntimeTests(unittest.TestCase):
    """Validate compute-node runtime behavior after discovery succeeds."""

    @mock.patch("builtins.print")
    @mock.patch("compute_node.runtime.load_runtime_processor_inventory")
    @mock.patch("compute_node.runtime.collect_hardware_profile")
    def test_run_registers_records_heartbeat_and_executes_task(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        hardware = HardwareProfile(
            hostname="worker-a",
            local_ip="10.0.0.2",
            mac_address="aa:bb:cc:dd:ee:ff",
            system="Windows",
            release="11",
            machine="AMD64",
            processor="x86_64",
            logical_cpu_count=12,
            memory_bytes=17179869184,
        )
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(hardware_type="cuda", effective_gflops=125.0, rank=1, best_config={"block_size": 256, "tile_size": 1, "transpose": False}),
            )
        )
        collect_hardware_profile_mock.return_value = hardware
        load_runtime_processor_inventory_mock.return_value = inventory
        task_assign = RuntimeEnvelope(
            kind=MessageKind.TASK_ASSIGN,
            task_assign=TaskAssign(
                request_id="req-1",
                node_id=COMPUTE_NODE_NAME,
                task_id="task-1",
                method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
                object_id="input_matrix/default",
                stream_id="stream-1",
                timestamp_ms=123456,
                row_start=0,
                row_end=2,
                vector_length=4,
                vector_data=pack_float32_values([1.0, 2.0, 3.0, 4.0]),
                iteration_count=5,
            ),
        )
        fake_session = _FakeSession(
            messages=[
                RuntimeEnvelope(
                    kind=MessageKind.HEARTBEAT,
                    heartbeat=Heartbeat(main_node_name=MAIN_NODE_NAME, unix_time_ms=123455),
                ),
                task_assign,
                None,
            ],
            register_ok=RegisterOk(
                main_node_name=MAIN_NODE_NAME,
                main_node_ip="10.0.0.5",
                main_node_port=52020,
                node_id="worker-1",
            ),
        )
        fake_executor = _FakeExecutor(inventory)

        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: fake_session,
            task_executor_factory=lambda runtime_inventory: fake_executor,
        )

        result = runtime.run()

        self.assertEqual(
            result,
            DiscoveryResult(
                success=False,
                peer_address="10.0.0.5",
                peer_port=52020,
                source="compute_node",
                message="Main node closed the TCP session.",
            ),
        )
        self.assertTrue(fake_session.connected)
        self.assertTrue(fake_session.closed)
        self.assertEqual(fake_session.register_args[0], COMPUTE_NODE_NAME)
        self.assertEqual(fake_session.register_args[1], hardware)
        self.assertEqual(fake_session.register_args[2].hardware_count, 1)
        self.assertEqual(len(fake_session.sent_messages), 3)
        self.assertEqual(fake_session.sent_messages[0].kind, MessageKind.HEARTBEAT_OK)
        self.assertEqual(fake_session.sent_messages[1].kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(fake_session.sent_messages[2].kind, MessageKind.TASK_RESULT)
        self.assertEqual(fake_session.sent_messages[1].task_accept.node_id, "worker-1")
        self.assertEqual(fake_session.sent_messages[2].task_result.node_id, "worker-1")
        self.assertEqual(fake_session.sent_messages[2].task_result.iteration_count, 5)
        self.assertEqual(fake_session.sent_messages[2].task_result.output_length, 2)
        self.assertEqual(len(fake_executor.tasks), 1)
        self.assertTrue(print_mock.called)

    @mock.patch("compute_node.runtime.load_runtime_processor_inventory")
    @mock.patch("compute_node.runtime.collect_hardware_profile")
    def test_run_reports_registration_failure(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
    ) -> None:
        collect_hardware_profile_mock.return_value = HardwareProfile(
            hostname="worker-a",
            local_ip="10.0.0.2",
            mac_address="aa:bb:cc:dd:ee:ff",
            system="Windows",
            release="11",
            machine="AMD64",
            processor="x86_64",
            logical_cpu_count=12,
            memory_bytes=17179869184,
        )
        load_runtime_processor_inventory_mock.return_value = RuntimeProcessorInventory(
            processors=(RuntimeProcessorProfile(hardware_type="cpu", effective_gflops=24.0, rank=1, best_config={"workers": 8, "tile_size": 512}),)
        )

        class _BrokenSession(_FakeSession):
            def connect(self) -> None:
                raise OSError("connect failed")

        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: _BrokenSession([], RegisterOk("", "", 0, "")),
        )

        result = runtime.run()

        self.assertFalse(result.success)
        self.assertIn("Unable to join main-node TCP runtime", result.message)


if __name__ == "__main__":
    unittest.main()


