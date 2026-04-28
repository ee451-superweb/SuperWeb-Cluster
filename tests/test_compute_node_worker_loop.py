"""Compute-node runtime tests."""

import threading
import unittest
from unittest import mock

import compute_node.worker_loop as runtime_module
from core.float32_codec import pack_float32_values
from core.types import ComputePerformanceSummary, DiscoveryResult, HardwareProfile
from compute_node.performance_metrics.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile
from compute_node.worker_loop import ComputeNodeRuntime
from core.config import AppConfig
from core.constants import (
    COMPUTE_NODE_NAME,
    MAIN_NODE_NAME,
    METHOD_GEMV,
    STATUS_OK,
)
from wire.internal_protocol.transport import (
    GemvResultPayload,
    Heartbeat,
    MessageKind,
    RegisterOk,
    RuntimeEnvelope,
    TaskAssign,
    TaskResult,
)


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


class _BlockingExecutor:
    def __init__(self, release_event: threading.Event) -> None:
        self.release_event = release_event
        self.started = threading.Event()
        self.tasks = []

    def execute_task(self, task) -> TaskResult:
        self.tasks.append(task)
        self.started.set()
        self.release_event.wait(timeout=1.0)
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


class _TaskThenHeartbeatSession(_FakeSession):
    def __init__(
        self,
        messages: list[RuntimeEnvelope | None],
        register_ok: RegisterOk,
        *,
        task_started: threading.Event,
        release_event: threading.Event,
    ) -> None:
        super().__init__(messages, register_ok)
        self._task_started = task_started
        self._release_event = release_event

    def receive(self):
        message = self.messages.pop(0)
        if message is not None and message.kind == MessageKind.HEARTBEAT:
            self._task_started.wait(timeout=1.0)
        if message is None:
            self._release_event.set()
        return message


class ComputeNodeRuntimeTests(unittest.TestCase):
    """Validate compute-node runtime behavior after discovery succeeds."""

    @mock.patch("builtins.print")
    @mock.patch("compute_node.worker_loop.write_audit_event")
    @mock.patch("compute_node.worker_loop.load_compute_performance_summary")
    @mock.patch("compute_node.worker_loop.load_runtime_processor_inventory")
    @mock.patch("compute_node.worker_loop.collect_hardware_profile")
    def test_run_registers_records_heartbeat_and_executes_task(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        load_compute_performance_summary_mock: mock.Mock,
        write_audit_event_mock: mock.Mock,
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
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary()
        task_assign = RuntimeEnvelope(
            kind=MessageKind.TASK_ASSIGN,
            task_assign=TaskAssign(
                request_id="req-1",
                node_id=COMPUTE_NODE_NAME,
                task_id="task-1",
                method=METHOD_GEMV,
                size="small",
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
        self.assertIsInstance(fake_session.sent_messages[2].task_result.result_payload, GemvResultPayload)
        self.assertEqual(len(fake_executor.tasks), 1)
        self.assertTrue(print_mock.called)
        self.assertTrue(
            any(
                "started as compute node" in call.args[0]
                for call in write_audit_event_mock.call_args_list
            )
        )

    @mock.patch("builtins.print")
    @mock.patch("compute_node.worker_loop.load_compute_performance_summary")
    @mock.patch("compute_node.worker_loop.load_runtime_processor_inventory")
    @mock.patch("compute_node.worker_loop.collect_hardware_profile")
    def test_run_replies_to_heartbeat_while_task_is_still_running(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        load_compute_performance_summary_mock: mock.Mock,
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
                RuntimeProcessorProfile(
                    hardware_type="cpu",
                    effective_gflops=24.0,
                    rank=1,
                    best_config={"workers": 8, "tile_size": 512},
                ),
            )
        )
        collect_hardware_profile_mock.return_value = hardware
        load_runtime_processor_inventory_mock.return_value = inventory
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary()
        task_assign = RuntimeEnvelope(
            kind=MessageKind.TASK_ASSIGN,
            task_assign=TaskAssign(
                request_id="req-1",
                node_id=COMPUTE_NODE_NAME,
                task_id="task-1",
                method=METHOD_GEMV,
                size="small",
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
        release_event = threading.Event()
        blocking_executor = _BlockingExecutor(release_event)
        fake_session = _TaskThenHeartbeatSession(
            messages=[
                task_assign,
                RuntimeEnvelope(
                    kind=MessageKind.HEARTBEAT,
                    heartbeat=Heartbeat(main_node_name=MAIN_NODE_NAME, unix_time_ms=123457),
                ),
                None,
            ],
            register_ok=RegisterOk(
                main_node_name=MAIN_NODE_NAME,
                main_node_ip="10.0.0.5",
                main_node_port=52020,
                node_id="worker-1",
            ),
            task_started=blocking_executor.started,
            release_event=release_event,
        )

        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: fake_session,
            task_executor_factory=lambda runtime_inventory: blocking_executor,
        )

        result = runtime.run()

        self.assertFalse(result.success)
        self.assertEqual(len(fake_session.sent_messages), 2)
        self.assertEqual(fake_session.sent_messages[0].kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(fake_session.sent_messages[1].kind, MessageKind.HEARTBEAT_OK)
        self.assertEqual(fake_session.sent_messages[1].heartbeat_ok.active_task_ids, ("task-1",))
        self.assertEqual(len(blocking_executor.tasks), 1)
        self.assertTrue(print_mock.called)

    @mock.patch("compute_node.worker_loop.load_compute_performance_summary")
    @mock.patch("compute_node.worker_loop.load_runtime_processor_inventory")
    @mock.patch("compute_node.worker_loop.collect_hardware_profile")
    def test_run_reports_registration_failure(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        load_compute_performance_summary_mock: mock.Mock,
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
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary()

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

    @mock.patch("compute_node.worker_loop.multiprocessing.get_context")
    @mock.patch("compute_node.worker_loop.ProcessPoolExecutor")
    def test_build_task_execution_backend_uses_sigint_ignoring_initializer(
        self,
        process_pool_executor_mock: mock.Mock,
        get_context_mock: mock.Mock,
    ) -> None:
        get_context_mock.return_value = object()
        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
        )

        runtime._build_task_execution_backend()

        process_pool_executor_mock.assert_called_once()
        self.assertEqual(
            process_pool_executor_mock.call_args.kwargs["initializer"],
            runtime_module._configure_subprocess_worker_signals,
        )

    @mock.patch("compute_node.worker_loop.load_compute_performance_summary")
    @mock.patch("compute_node.worker_loop.load_runtime_processor_inventory")
    @mock.patch("compute_node.worker_loop.collect_hardware_profile")
    def test_run_waits_for_idle_process_pool_shutdown(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        load_compute_performance_summary_mock: mock.Mock,
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
            processors=(RuntimeProcessorProfile(hardware_type="cpu", effective_gflops=24.0, rank=1, best_config={}),)
        )
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary()
        fake_session = _FakeSession(
            messages=[None],
            register_ok=RegisterOk(
                main_node_name=MAIN_NODE_NAME,
                main_node_ip="10.0.0.5",
                main_node_port=52020,
                node_id="worker-1",
            ),
        )
        fake_pool = mock.Mock()
        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: fake_session,
        )
        runtime._build_task_execution_backend = mock.Mock(return_value=(fake_pool, None, None))

        runtime.run()

        fake_pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    @mock.patch("compute_node.worker_loop.load_compute_performance_summary")
    @mock.patch("compute_node.worker_loop.load_runtime_processor_inventory")
    @mock.patch("compute_node.worker_loop.collect_hardware_profile")
    def test_run_terminates_inflight_process_pool_workers_on_shutdown(
        self,
        collect_hardware_profile_mock: mock.Mock,
        load_runtime_processor_inventory_mock: mock.Mock,
        load_compute_performance_summary_mock: mock.Mock,
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
            processors=(RuntimeProcessorProfile(hardware_type="cpu", effective_gflops=24.0, rank=1, best_config={}),)
        )
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary()
        task_assign = RuntimeEnvelope(
            kind=MessageKind.TASK_ASSIGN,
            task_assign=TaskAssign(
                request_id="req-1",
                node_id=COMPUTE_NODE_NAME,
                task_id="task-1",
                method=METHOD_GEMV,
                size="small",
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
            messages=[task_assign, None],
            register_ok=RegisterOk(
                main_node_name=MAIN_NODE_NAME,
                main_node_ip="10.0.0.5",
                main_node_port=52020,
                node_id="worker-1",
            ),
        )
        running_future = mock.Mock()
        running_future.done.return_value = False
        worker_process = mock.Mock()
        worker_process.is_alive.return_value = True
        fake_pool = mock.Mock()
        fake_pool.submit.return_value = running_future
        fake_pool._processes = {1234: worker_process}
        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: fake_session,
        )
        runtime._build_task_execution_backend = mock.Mock(return_value=(fake_pool, None, None))

        runtime.run()

        fake_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        worker_process.terminate.assert_called_once()
        worker_process.join.assert_called_once_with(timeout=1.0)


class SubprocessEntrypointPinnedBackendTests(unittest.TestCase):
    """Validate that the subprocess task entrypoint forwards pinned_backend."""

    def test_execute_task_in_subprocess_forwards_pinned_backend(self) -> None:
        captured_kwargs: dict[str, object] = {}

        fake_router = mock.Mock()
        fake_router.execute_task.return_value = "result-sentinel"
        fake_registry = mock.Mock()

        def _fake_build_handlers(**kwargs):
            captured_kwargs.update(kwargs)
            return fake_registry

        with (
            mock.patch.object(runtime_module, "build_default_method_handlers", side_effect=_fake_build_handlers),
            mock.patch.object(runtime_module, "TaskExecutionRouter", return_value=fake_router),
        ):
            result = runtime_module._execute_task_in_subprocess(
                task=mock.sentinel.task,
                pinned_backend="cpu",
            )

        self.assertEqual(result, "result-sentinel")
        self.assertEqual(captured_kwargs, {"pinned_backend": "cpu"})
        fake_router.close.assert_called_once_with()

    def test_execute_task_in_subprocess_defaults_to_none_backend(self) -> None:
        captured_kwargs: dict[str, object] = {}
        fake_router = mock.Mock()
        fake_router.execute_task.return_value = "ok"

        def _fake_build_handlers(**kwargs):
            captured_kwargs.update(kwargs)
            return mock.Mock()

        with (
            mock.patch.object(runtime_module, "build_default_method_handlers", side_effect=_fake_build_handlers),
            mock.patch.object(runtime_module, "TaskExecutionRouter", return_value=fake_router),
        ):
            runtime_module._execute_task_in_subprocess(task=mock.sentinel.task)

        self.assertEqual(captured_kwargs, {"pinned_backend": None})


if __name__ == "__main__":
    unittest.main()
