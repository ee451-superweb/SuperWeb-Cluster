"""Main node runtime tests."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from core.float32_codec import pack_float32_values, unpack_float32_bytes
from core.types import ComputeHardwarePerformance, ComputePerformanceSummary, DiscoveryResult, HardwareProfile
from core.config import AppConfig
from core.constants import (
    COMPUTE_NODE_NAME,
    DEFAULT_TCP_PORT,
    MAIN_NODE_NAME,
    METHOD_FREE_CONTENT,
    METHOD_GEMV,
    METHOD_CONV2D,
    STATUS_ACCEPTED,
    STATUS_OK,
    SUPERWEB_CLIENT_NAME,
)
from main_node.dispatcher import WorkerTaskSlice
from main_node.mailbox import RuntimeConnectionMailbox
from main_node.control_loop import MainNodeRuntime
from wire.discovery_protocol import build_discover_message
from wire.internal_protocol.transport import (
    ArtifactDescriptor,
    GemvTaskPayload,
    Heartbeat,
    MessageKind,
    RuntimeEnvelope,
    Conv2dResultPayload,
    TaskAccept,
    TaskFail,
    TaskResult,
    TransferMode,
    build_client_join,
    build_client_request,
    build_heartbeat,
    build_heartbeat_ok,
    build_register_worker,
)


class MainNodeRuntimeTests(unittest.TestCase):
    """Validate the promoted main-node loop."""

    @mock.patch("main_node.control_loop.write_audit_event")
    @mock.patch("main_node.control_loop.network.safe_close")
    @mock.patch("main_node.control_loop.MainNodeRuntime._close_runtime_connections")
    @mock.patch("main_node.control_loop.threading.Thread")
    @mock.patch("main_node.control_loop.MainNodeRuntime._create_tcp_listener")
    @mock.patch("main_node.control_loop.multicast.close")
    @mock.patch("main_node.control_loop.multicast.recv_packet")
    @mock.patch("main_node.control_loop.network.get_local_mac_address")
    @mock.patch("main_node.control_loop.network.resolve_local_ip")
    @mock.patch("main_node.control_loop.network.set_socket_timeout")
    @mock.patch("main_node.control_loop.multicast.create_receiver")
    def test_run_starts_loop_and_returns_success_on_shutdown(
        self,
        create_receiver_mock: mock.Mock,
        set_socket_timeout_mock: mock.Mock,
        resolve_local_ip_mock: mock.Mock,
        get_local_mac_address_mock: mock.Mock,
        recv_packet_mock: mock.Mock,
        close_mock: mock.Mock,
        create_tcp_listener_mock: mock.Mock,
        thread_mock: mock.Mock,
        close_runtime_connections_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        write_audit_event_mock: mock.Mock,
    ) -> None:
        endpoint = mock.Mock()
        endpoint.sock = mock.Mock()
        create_receiver_mock.return_value = endpoint
        runtime_sock = mock.Mock()
        create_tcp_listener_mock.return_value = runtime_sock
        resolve_local_ip_mock.return_value = "10.0.0.5"
        get_local_mac_address_mock.return_value = "aa:bb:cc:dd:ee:ff"
        recv_packet_mock.return_value = (("10.0.0.2", 5000), build_discover_message(COMPUTE_NODE_NAME))
        thread_instance = mock.Mock()
        thread_mock.return_value = thread_instance

        stop_values = iter([False, True])
        runtime = MainNodeRuntime(
            config=AppConfig(),
            logger=mock.Mock(),
            should_stop=lambda: next(stop_values),
        )
        runtime.artifact_manager = mock.Mock()
        runtime.artifact_manager.port = 52021

        result = runtime.run()

        self.assertEqual(
            result,
            DiscoveryResult(
                success=True,
                peer_address="10.0.0.5",
                peer_port=DEFAULT_TCP_PORT,
                source="main_node",
                message="Main-node runtime stopped.",
            ),
        )
        set_socket_timeout_mock.assert_called_once()
        create_tcp_listener_mock.assert_called_once()
        self.assertEqual(thread_mock.call_count, 2)
        self.assertEqual(thread_instance.start.call_count, 2)
        recv_packet_mock.assert_called_once()
        safe_close_mock.assert_called_once_with(runtime_sock)
        close_runtime_connections_mock.assert_called_once()
        close_mock.assert_called_once_with(endpoint)
        runtime.logger.info.assert_called()
        self.assertTrue(
            any(
                "started as main node" in call.args[0]
                for call in write_audit_event_mock.call_args_list
            )
        )

    @mock.patch("main_node.control_loop.multicast.send_announce")
    def test_handle_packet_replies_to_discover(
        self,
        send_announce_mock: mock.Mock,
    ) -> None:
        send_announce_mock.return_value = "10.0.0.5"
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )

        runtime._handle_packet(
            mock.Mock(),
            ("10.0.0.2", 5000),
            build_discover_message(COMPUTE_NODE_NAME),
        )

        send_announce_mock.assert_called_once()
        runtime.logger.info.assert_called()

    @mock.patch("builtins.print")
    @mock.patch("main_node.control_loop.MainNodeRuntime._start_runtime_connection_reader")
    @mock.patch("main_node.control_loop.send_message")
    @mock.patch("main_node.control_loop.recv_message")
    def test_register_runtime_connection_accepts_register_worker(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        start_runtime_connection_reader_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.total_registered_gflops.return_value = 149.0
        runtime.registry.count_workers.return_value = 1
        runtime.registry.count_registered_hardware.return_value = 2
        worker_connection = mock.Mock()
        worker_connection.node_name = COMPUTE_NODE_NAME
        worker_connection.runtime_id = "worker-1"
        worker_connection.peer_address = "10.0.0.2"
        worker_connection.peer_port = 5000
        worker_connection.hardware = hardware = HardwareProfile(
            hostname="worker-a",
            local_ip="10.0.0.2",
            mac_address="aa:bb:cc:dd:ee:ff",
            system="Windows",
            release="11",
            machine="AMD64",
            processor="x86_64",
            logical_cpu_count=8,
            memory_bytes=8589934592,
        )
        performance = ComputePerformanceSummary(
            hardware_count=2,
            ranked_hardware=[
                ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=125.0, rank=1),
                ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=2),
            ],
        )
        worker_connection.performance = performance
        runtime.registry.register_worker.return_value = worker_connection
        client_sock = mock.Mock()
        recv_message_mock.return_value = build_register_worker(COMPUTE_NODE_NAME, hardware, performance)

        runtime._register_runtime_connection(client_sock, ("10.0.0.2", 5000), "10.0.0.5")

        runtime.registry.register_worker.assert_called_once_with(
            node_name=COMPUTE_NODE_NAME,
            peer_address="10.0.0.2",
            peer_port=5000,
            hardware=hardware,
            performance=performance,
            sock=client_sock,
        )
        response = send_message_mock.call_args.args[1]
        self.assertEqual(response.kind, MessageKind.REGISTER_OK)
        self.assertEqual(response.register_ok.main_node_ip, "10.0.0.5")
        self.assertEqual(response.register_ok.node_id, "worker-1")
        start_runtime_connection_reader_mock.assert_called_once_with(worker_connection)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.control_loop.MainNodeRuntime._start_runtime_connection_reader")
    @mock.patch("main_node.control_loop.threading.Thread")
    @mock.patch("main_node.control_loop.send_message")
    @mock.patch("main_node.control_loop.recv_message")
    def test_register_runtime_connection_accepts_client_join(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        thread_mock: mock.Mock,
        start_runtime_connection_reader_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.total_registered_gflops.return_value = 0.0
        runtime.registry.count_registered_hardware.return_value = 0
        runtime.registry.count_workers.return_value = 2
        runtime.registry.count_clients.return_value = 1
        client_connection = mock.Mock()
        client_connection.node_name = SUPERWEB_CLIENT_NAME
        client_connection.runtime_id = "client-1"
        runtime.registry.register_client.return_value = client_connection
        thread_instance = mock.Mock()
        thread_mock.return_value = thread_instance
        client_sock = mock.Mock()
        recv_message_mock.return_value = build_client_join(SUPERWEB_CLIENT_NAME)

        runtime._register_runtime_connection(client_sock, ("10.0.0.3", 6000), "10.0.0.5")

        runtime.registry.register_client.assert_called_once_with(
            node_name=SUPERWEB_CLIENT_NAME,
            peer_address="10.0.0.3",
            peer_port=6000,
            sock=client_sock,
        )
        response = send_message_mock.call_args.args[1]
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.request_id, "join")
        self.assertEqual(response.client_response.status_code, STATUS_OK)
        self.assertEqual(response.client_response.worker_count, 2)
        self.assertEqual(response.client_response.client_count, 1)
        self.assertEqual(response.client_response.client_id, "client-1")
        self.assertEqual(thread_mock.call_count, 1)
        self.assertEqual(thread_instance.start.call_count, 1)
        start_runtime_connection_reader_mock.assert_called_once_with(client_connection)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.control_loop.network.safe_close")
    @mock.patch("main_node.control_loop.send_message")
    @mock.patch("main_node.control_loop.recv_message")
    def test_serve_client_connection_dispatches_task_and_replies_with_aggregated_result(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = [mock.Mock()]
        runtime.registry.list_worker_hardware.return_value = [mock.Mock()]
        runtime.registry.count_workers.return_value = 1
        runtime.registry.count_clients.return_value = 1

        task_assignment = WorkerTaskSlice(
            connection=mock.Mock(
                node_name=COMPUTE_NODE_NAME,
                runtime_id="worker-1",
                peer_id="worker:compute node@10.0.0.2:5000",
                peer_address="10.0.0.2",
                peer_port=5000,
                sock=mock.Mock(),
                io_lock=threading.Lock(),
            ),
            task_id="req-1",
            artifact_id="req-1:worker-1",
            row_start=0,
            row_end=runtime.gemv_spec.rows,
            effective_gflops=125.0,
        )
        runtime.dispatcher = mock.Mock()
        runtime.dispatcher.dispatch_gemv.return_value = [task_assignment]
        task_result = TaskResult(
            request_id="req-1",
            node_id="worker-1",
            task_id="req-1:worker",
            timestamp_ms=123456,
            status_code=STATUS_OK,
            row_start=0,
            row_end=runtime.gemv_spec.rows,
            output_length=runtime.gemv_spec.rows,
            output_vector=pack_float32_values([1.0] * runtime.gemv_spec.rows),
            iteration_count=4,
        )
        from wire.internal_protocol.transport import WorkerTiming
        runtime._run_worker_task_slice = mock.Mock(
            return_value=(
                task_result,
                WorkerTiming(
                    node_id="worker-1",
                    task_id="req-1:worker",
                    slice="rows=0:0",
                    wall_ms=0,
                    artifact_fetch_ms=0,
                ),
            )
        )
        runtime.aggregator = mock.Mock()
        runtime.aggregator.collect_gemv_result.return_value = task_result.output_vector

        connection = mock.Mock()
        connection.peer_id = "client:superweb client@10.0.0.3:6000"
        connection.node_name = SUPERWEB_CLIENT_NAME
        connection.runtime_id = "client-1"
        connection.peer_address = "10.0.0.3"
        connection.peer_port = 6000
        connection.sock = mock.Mock()
        runtime.registry.allocate_task_id.return_value = "gemv-1"
        runtime.registry.remove_client.return_value = connection
        request_vector = pack_float32_values([1.0] * runtime.gemv_spec.cols)
        recv_message_mock.side_effect = [
            build_client_request(
                SUPERWEB_CLIENT_NAME,
                "req-1",
                METHOD_GEMV,
                request_vector,
                object_id="input_matrix/default",
                stream_id="stream-1",
                iteration_count=4,
            ),
            None,
        ]

        runtime._serve_client_connection(connection)

        runtime.registry.mark_client_request.assert_called_once_with(connection.peer_id)
        self.assertEqual(send_message_mock.call_count, 2)
        request_ok = send_message_mock.call_args_list[0].args[1]
        response = send_message_mock.call_args_list[1].args[1]
        self.assertEqual(request_ok.kind, MessageKind.CLIENT_REQUEST_OK)
        self.assertEqual(request_ok.client_request_ok.task_id, "gemv-1")
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.request_id, "gemv-1")
        self.assertEqual(response.client_response.task_id, "gemv-1")
        self.assertEqual(response.client_response.status_code, STATUS_OK)
        self.assertEqual(response.client_response.client_id, "client-1")
        self.assertEqual(response.client_response.iteration_count, 4)
        self.assertEqual(response.client_response.output_length, runtime.gemv_spec.rows)
        self.assertEqual(len(response.client_response.output_vector), runtime.gemv_spec.rows * 4)
        self.assertEqual(unpack_float32_bytes(response.client_response.output_vector)[:2], [1.0, 1.0])
        printed_messages = [call.args[0] for call in print_mock.call_args_list if call.args]
        self.assertIn(
            "CLIENT_RESPONSE to superweb client status_code=200 task_id=gemv-1",
            printed_messages,
        )
        self.assertTrue(
            all(
                "elapsed_ms=" not in message and "output_length=" not in message and "inline_bytes=" not in message
                for message in printed_messages
                if message.startswith("CLIENT_RESPONSE ")
            )
        )
        runtime.registry.remove_client.assert_called_once_with(connection.peer_id)
        safe_close_mock.assert_called_once_with(connection.sock)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.control_loop.network.safe_close")
    @mock.patch("main_node.control_loop.send_message")
    @mock.patch("main_node.control_loop.recv_message")
    def test_serve_client_connection_replies_to_free_content_with_system_overview(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.artifact_manager.set_public_host("10.0.0.5")
        runtime.registry = mock.Mock()
        runtime.registry.count_workers.return_value = 2
        runtime.registry.count_clients.return_value = 1
        runtime.registry.total_registered_gflops.return_value = 149.0
        runtime.registry.total_registered_gflops_by_method.return_value = {
            METHOD_GEMV: 125.0,
            METHOD_CONV2D: 24.0,
        }
        runtime.registry.allocate_task_id.return_value = "free-1"
        connection = mock.Mock()
        connection.peer_id = "client:superweb client@10.0.0.3:6000"
        connection.node_name = SUPERWEB_CLIENT_NAME
        connection.runtime_id = "client-1"
        connection.peer_address = "10.0.0.3"
        connection.peer_port = 6000
        connection.sock = mock.Mock()
        runtime.registry.remove_client.return_value = connection
        recv_message_mock.side_effect = [
            build_client_request(
                SUPERWEB_CLIENT_NAME,
                "req-1",
                METHOD_FREE_CONTENT,
                b"hello",
                iteration_count=1,
            ),
            None,
        ]

        runtime._serve_client_connection(connection)

        self.assertEqual(send_message_mock.call_count, 2)
        request_ok = send_message_mock.call_args_list[0].args[1]
        response = send_message_mock.call_args_list[1].args[1]
        self.assertEqual(request_ok.kind, MessageKind.CLIENT_REQUEST_OK)
        self.assertEqual(request_ok.client_request_ok.task_id, "free-1")
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.status_code, STATUS_OK)
        self.assertEqual(response.client_response.task_id, "free-1")
        self.assertEqual(response.client_response.method, METHOD_FREE_CONTENT)
        self.assertEqual(response.client_response.client_id, "client-1")
        overview_text = response.client_response.output_vector.decode("utf-8")
        self.assertIn("superweb-cluster system overview", overview_text)
        self.assertIn("main_node_endpoint: 10.0.0.5:52020", overview_text)
        self.assertIn("worker_count: 2", overview_text)
        self.assertIn("total_effective_gflops: 149.000", overview_text)
        self.assertIn("supported_methods: gemv, conv2d, gemm, free_content", overview_text)
        runtime.registry.remove_client.assert_called_once_with(connection.peer_id)
        safe_close_mock.assert_called_once_with(connection.sock)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.task_exchange.recv_message")
    @mock.patch("main_node.task_exchange.send_message")
    def test_run_worker_task_slice_exchanges_assign_accept_and_result(
        self,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        del print_mock
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )

        worker_connection = mock.Mock()
        worker_connection.node_name = COMPUTE_NODE_NAME
        worker_connection.runtime_id = "worker-1"
        worker_connection.peer_id = "worker:compute node@10.0.0.2:5000"
        worker_connection.peer_address = "10.0.0.2"
        worker_connection.peer_port = 5000
        worker_connection.sock = mock.Mock()
        worker_connection.sock.gettimeout.return_value = 1.0
        worker_connection.io_lock = threading.Lock()

        request = build_client_request(
            SUPERWEB_CLIENT_NAME,
            "req-1",
            METHOD_GEMV,
            pack_float32_values([1.0] * runtime.gemv_spec.cols),
            object_id="input_matrix/default",
            stream_id="stream-1",
            iteration_count=6,
        ).client_request
        assert request is not None

        recv_message_mock.side_effect = [
            RuntimeEnvelope(
                kind=MessageKind.TASK_ACCEPT,
                task_accept=TaskAccept(
                    request_id="req-1",
                    node_id="worker-1",
                    task_id="req-1",
                    timestamp_ms=123456,
                    status_code=STATUS_ACCEPTED,
                ),
            ),
            RuntimeEnvelope(
                kind=MessageKind.TASK_RESULT,
                task_result=TaskResult(
                    request_id="req-1",
                    node_id="worker-1",
                    task_id="req-1",
                    timestamp_ms=123457,
                    status_code=STATUS_OK,
                    row_start=0,
                    row_end=10,
                    output_length=10,
                    output_vector=pack_float32_values([1.0] * 10),
                    iteration_count=6,
                ),
            ),
        ]

        assignment = WorkerTaskSlice(
            connection=worker_connection,
            task_id="req-1",
            artifact_id="req-1:worker-1",
            row_start=0,
            row_end=10,
            effective_gflops=125.0,
        )

        result, timing = runtime._run_worker_task_slice(request, assignment)

        self.assertEqual(result.task_id, "req-1")
        self.assertEqual(result.row_end, 10)
        self.assertEqual(timing.task_id, "req-1")
        self.assertEqual(timing.slice, "rows=0:10")
        sent_message = send_message_mock.call_args.args[1]
        self.assertEqual(sent_message.kind, MessageKind.TASK_ASSIGN)
        self.assertEqual(sent_message.task_assign.node_id, "worker-1")
        self.assertEqual(sent_message.task_assign.artifact_id, "req-1:worker-1")
        self.assertEqual(sent_message.task_assign.row_start, 0)
        self.assertEqual(sent_message.task_assign.row_end, 10)
        self.assertEqual(sent_message.task_assign.iteration_count, 6)
        self.assertEqual(sent_message.task_assign.transfer_mode, TransferMode.INLINE_PREFERRED)
        self.assertIsInstance(sent_message.task_assign.task_payload, GemvTaskPayload)

    @mock.patch("builtins.print")
    @mock.patch("main_node.task_exchange.recv_message")
    @mock.patch("main_node.task_exchange.send_message")
    def test_run_worker_task_slice_fetches_artifact_back_into_result_payload(
        self,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        del print_mock
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.artifact_manager._server.port = 0
        self.addCleanup(runtime.artifact_manager.close)

        worker_connection = mock.Mock()
        worker_connection.node_name = COMPUTE_NODE_NAME
        worker_connection.runtime_id = "worker-1"
        worker_connection.peer_id = "worker:compute node@10.0.0.2:5000"
        worker_connection.peer_address = "10.0.0.2"
        worker_connection.peer_port = 5000
        worker_connection.sock = mock.Mock()
        worker_connection.sock.gettimeout.return_value = 1.0
        worker_connection.io_lock = threading.Lock()

        request = build_client_request(
            SUPERWEB_CLIENT_NAME,
            "req-spatial",
            METHOD_CONV2D,
            b"",
            size="small",
            object_id="conv2d/small",
            stream_id="stream-spatial",
            iteration_count=1,
        ).client_request
        assert request is not None

        recv_message_mock.side_effect = [
            RuntimeEnvelope(
                kind=MessageKind.TASK_ACCEPT,
                task_accept=TaskAccept(
                    request_id="req-spatial",
                    node_id="worker-1",
                    task_id="req-spatial",
                    timestamp_ms=123456,
                    status_code=STATUS_ACCEPTED,
                ),
            ),
            RuntimeEnvelope(
                kind=MessageKind.TASK_RESULT,
                task_result=TaskResult(
                    request_id="req-spatial",
                    node_id="worker-1",
                    task_id="req-spatial",
                    timestamp_ms=123457,
                    status_code=STATUS_OK,
                    iteration_count=1,
                    result_payload=Conv2dResultPayload(
                        start_oc=0,
                        end_oc=2,
                        output_h=8,
                        output_w=8,
                        output_length=128,
                        output_vector=b"",
                        result_artifact_id="artifact-1",
                    ),
                    result_artifact=ArtifactDescriptor(
                        artifact_id="artifact-1",
                        content_type="application/octet-stream",
                        size_bytes=512,
                        checksum="abc123",
                        producer_node_id="worker-1",
                        transfer_host="127.0.0.1",
                        transfer_port=53000,
                        chunk_size=262144,
                        ready=True,
                    ),
                ),
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime.artifact_manager.root_dir = Path(temp_dir)

            def fake_fetch_to_file(descriptor, destination_path, *, timeout):
                del descriptor, timeout
                destination_path.write_bytes(pack_float32_values([1.0] * 128))
                return destination_path

            runtime.artifact_manager.fetch_to_file = mock.Mock(side_effect=fake_fetch_to_file)

            assignment = WorkerTaskSlice(
                connection=worker_connection,
                task_id="req-spatial",
                artifact_id="artifact-1",
                row_start=0,
                row_end=0,
                effective_gflops=125.0,
                method="conv2d",
                start_oc=0,
                end_oc=2,
            )

            result, timing = runtime._run_worker_task_slice(request, assignment)

        runtime.artifact_manager.fetch_to_file.assert_called_once()
        self.assertEqual(result.output_length, 128)
        self.assertEqual(result.output_vector, b"")
        self.assertTrue(result.local_result_path)
        self.assertTrue(Path(result.local_result_path).name.startswith("fetch-artifact-1-"))
        self.assertEqual(result.start_oc, 0)
        self.assertEqual(result.end_oc, 2)
        self.assertEqual(timing.slice, "oc=0:2")
        self.assertEqual(send_message_mock.call_count, 2)
        assign_message = send_message_mock.call_args_list[0].args[1]
        release_message = send_message_mock.call_args_list[1].args[1]
        self.assertEqual(assign_message.task_assign.transfer_mode, TransferMode.ARTIFACT_REQUIRED)
        self.assertEqual(release_message.kind, MessageKind.ARTIFACT_RELEASE)
        self.assertEqual(release_message.artifact_release.artifact_id, "artifact-1")
        weight_artifact = assign_message.task_assign.task_payload.weight_artifact
        self.assertIsNotNone(weight_artifact)
        self.assertEqual(weight_artifact.artifact_id, "artifact-1-weight")
        self.assertEqual(
            weight_artifact.content_type, "application/x-superweb-conv2d-weight"
        )
        self.assertEqual(assign_message.task_assign.task_payload.weight_data, b"")
        self.assertNotIn(weight_artifact.artifact_id, runtime.artifact_manager._artifacts)

    @mock.patch("builtins.print")
    @mock.patch("main_node.task_exchange.recv_message")
    @mock.patch("main_node.task_exchange.send_message")
    def test_run_worker_task_slice_cleans_up_weight_artifact_on_task_fail(
        self,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        del send_message_mock, print_mock
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.task_exchange._remove_worker_connection = mock.Mock()
        runtime.artifact_manager._server.port = 0
        self.addCleanup(runtime.artifact_manager.close)

        worker_connection = mock.Mock()
        worker_connection.node_name = COMPUTE_NODE_NAME
        worker_connection.runtime_id = "worker-1"
        worker_connection.peer_id = "worker:compute node@10.0.0.2:5000"
        worker_connection.peer_address = "10.0.0.2"
        worker_connection.peer_port = 5000
        worker_connection.sock = mock.Mock()
        worker_connection.sock.gettimeout.return_value = 1.0
        worker_connection.io_lock = threading.Lock()

        request = build_client_request(
            SUPERWEB_CLIENT_NAME,
            "req-fail",
            METHOD_CONV2D,
            b"",
            size="small",
            object_id="conv2d/small",
            stream_id="stream-fail",
            iteration_count=1,
        ).client_request
        assert request is not None

        recv_message_mock.side_effect = [
            RuntimeEnvelope(
                kind=MessageKind.TASK_ACCEPT,
                task_accept=TaskAccept(
                    request_id="req-fail",
                    node_id="worker-1",
                    task_id="req-fail",
                    timestamp_ms=123456,
                    status_code=STATUS_ACCEPTED,
                ),
            ),
            RuntimeEnvelope(
                kind=MessageKind.TASK_FAIL,
                task_fail=TaskFail(
                    request_id="req-fail",
                    node_id="worker-1",
                    task_id="req-fail",
                    timestamp_ms=123457,
                    status_code=STATUS_OK,
                    error_message="worker boom",
                ),
            ),
        ]

        assignment = WorkerTaskSlice(
            connection=worker_connection,
            task_id="req-fail",
            artifact_id="req-fail:worker-1:0",
            row_start=0,
            row_end=0,
            effective_gflops=125.0,
            method="conv2d",
            start_oc=0,
            end_oc=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime.artifact_manager.root_dir = Path(temp_dir)
            with self.assertRaises(RuntimeError) as ctx:
                runtime._run_worker_task_slice(request, assignment)

            self.assertIn("worker boom", str(ctx.exception))
            self.assertNotIn(
                "req-fail:worker-1:0-weight", runtime.artifact_manager._artifacts
            )
        runtime.task_exchange._remove_worker_connection.assert_called_once()

    @mock.patch("builtins.print")
    @mock.patch("main_node.task_exchange.recv_message")
    @mock.patch("main_node.task_exchange.send_message")
    def test_run_worker_task_slice_uses_unique_weight_artifact_per_worker_slice(
        self,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        del print_mock
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.task_exchange._remove_worker_connection = mock.Mock()
        runtime.artifact_manager._server.port = 0
        self.addCleanup(runtime.artifact_manager.close)

        def _worker_fail(task_id: str, node_id: str) -> list[RuntimeEnvelope]:
            return [
                RuntimeEnvelope(
                    kind=MessageKind.TASK_ACCEPT,
                    task_accept=TaskAccept(
                        request_id="req-shared",
                        node_id=node_id,
                        task_id=task_id,
                        timestamp_ms=123456,
                        status_code=STATUS_ACCEPTED,
                    ),
                ),
                RuntimeEnvelope(
                    kind=MessageKind.TASK_FAIL,
                    task_fail=TaskFail(
                        request_id="req-shared",
                        node_id=node_id,
                        task_id=task_id,
                        timestamp_ms=123457,
                        status_code=STATUS_OK,
                        error_message="worker boom",
                    ),
                ),
            ]

        recv_message_mock.side_effect = (
            _worker_fail("req-shared", "worker-1")
            + _worker_fail("req-shared", "worker-2")
        )

        request = build_client_request(
            SUPERWEB_CLIENT_NAME,
            "req-shared",
            METHOD_CONV2D,
            b"",
            size="small",
            object_id="conv2d/small",
            stream_id="stream-shared",
            iteration_count=1,
        ).client_request
        assert request is not None

        worker_1 = mock.Mock()
        worker_1.node_name = COMPUTE_NODE_NAME
        worker_1.runtime_id = "worker-1"
        worker_1.peer_id = "worker:compute node@10.0.0.2:5000"
        worker_1.peer_address = "10.0.0.2"
        worker_1.peer_port = 5000
        worker_1.sock = mock.Mock()
        worker_1.sock.gettimeout.return_value = 1.0
        worker_1.io_lock = threading.Lock()

        worker_2 = mock.Mock()
        worker_2.node_name = COMPUTE_NODE_NAME
        worker_2.runtime_id = "worker-2"
        worker_2.peer_id = "worker:compute node@10.0.0.3:5000"
        worker_2.peer_address = "10.0.0.3"
        worker_2.peer_port = 5000
        worker_2.sock = mock.Mock()
        worker_2.sock.gettimeout.return_value = 1.0
        worker_2.io_lock = threading.Lock()

        assignment_1 = WorkerTaskSlice(
            connection=worker_1,
            task_id="req-shared",
            artifact_id="req-shared:worker-1:0",
            row_start=0,
            row_end=0,
            effective_gflops=125.0,
            method="conv2d",
            start_oc=0,
            end_oc=1,
        )
        assignment_2 = WorkerTaskSlice(
            connection=worker_2,
            task_id="req-shared",
            artifact_id="req-shared:worker-2:0",
            row_start=0,
            row_end=0,
            effective_gflops=125.0,
            method="conv2d",
            start_oc=1,
            end_oc=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime.artifact_manager.root_dir = Path(temp_dir)
            with self.assertRaises(RuntimeError):
                runtime._run_worker_task_slice(request, assignment_1)
            with self.assertRaises(RuntimeError):
                runtime._run_worker_task_slice(request, assignment_2)

        sent_assign_messages = [call.args[1] for call in send_message_mock.call_args_list]
        self.assertEqual(len(sent_assign_messages), 2)
        first_weight_artifact = sent_assign_messages[0].task_assign.task_payload.weight_artifact
        second_weight_artifact = sent_assign_messages[1].task_assign.task_payload.weight_artifact
        assert first_weight_artifact is not None
        assert second_weight_artifact is not None
        self.assertEqual(first_weight_artifact.artifact_id, "req-shared:worker-1:0-weight")
        self.assertEqual(second_weight_artifact.artifact_id, "req-shared:worker-2:0-weight")
        self.assertNotEqual(first_weight_artifact.artifact_id, second_weight_artifact.artifact_id)

    @mock.patch("builtins.print")
    @mock.patch("main_node.task_exchange.send_message")
    def test_run_worker_task_slice_waits_through_mailbox_without_result_deadline(
        self,
        send_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        del print_mock
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.artifact_manager._server.port = 0
        self.addCleanup(runtime.artifact_manager.close)

        worker_connection = mock.Mock()
        worker_connection.node_name = COMPUTE_NODE_NAME
        worker_connection.runtime_id = "worker-1"
        worker_connection.peer_id = "worker:compute node@10.0.0.2:5000"
        worker_connection.peer_address = "10.0.0.2"
        worker_connection.peer_port = 5000
        worker_connection.sock = mock.Mock()
        worker_connection.io_lock = threading.Lock()
        worker_connection.task_lock = threading.Lock()
        worker_connection.mailbox = RuntimeConnectionMailbox()
        worker_connection.mailbox.wait_for_task_message = mock.Mock(side_effect=[
            RuntimeEnvelope(
                kind=MessageKind.TASK_ACCEPT,
                task_accept=TaskAccept(
                    request_id="req-spatial-runtime",
                    node_id="worker-1",
                    task_id="req-spatial-runtime",
                    timestamp_ms=123456,
                    status_code=STATUS_ACCEPTED,
                ),
            ),
            RuntimeEnvelope(
                kind=MessageKind.TASK_RESULT,
                task_result=TaskResult(
                    request_id="req-spatial-runtime",
                    node_id="worker-1",
                    task_id="req-spatial-runtime",
                    timestamp_ms=123457,
                    status_code=STATUS_OK,
                    iteration_count=1,
                    result_payload=Conv2dResultPayload(
                        start_oc=0,
                        end_oc=2,
                        output_h=8,
                        output_w=8,
                        output_length=128,
                        output_vector=pack_float32_values([1.0] * 128),
                    ),
                ),
            ),
        ])

        request = build_client_request(
            SUPERWEB_CLIENT_NAME,
            "req-spatial-runtime",
            METHOD_CONV2D,
            b"",
            size="small",
            object_id="conv2d/small",
            stream_id="stream-spatial-runtime",
            iteration_count=1,
        ).client_request
        assert request is not None

        assignment = WorkerTaskSlice(
            connection=worker_connection,
            task_id="req-spatial-runtime",
            artifact_id="req-spatial-runtime:worker-1",
            row_start=0,
            row_end=0,
            effective_gflops=125.0,
            method="conv2d",
            start_oc=0,
            end_oc=2,
        )

        result, _timing = runtime._run_worker_task_slice(request, assignment)

        self.assertEqual(result.task_id, "req-spatial-runtime")
        mailbox_calls = worker_connection.mailbox.wait_for_task_message.call_args_list
        self.assertEqual(mailbox_calls[0].kwargs["timeout"], None)
        self.assertEqual(mailbox_calls[1].kwargs["timeout"], None)
        self.assertEqual(send_message_mock.call_count, 1)

    @mock.patch("main_node.control_loop.network.safe_close")
    @mock.patch("main_node.heartbeat.send_message")
    def test_send_heartbeat_once_retries_and_removes_timed_out_worker(
        self,
        send_message_mock: mock.Mock,
        safe_close_mock: mock.Mock,
    ) -> None:
        send_message_mock.side_effect = OSError("broken socket")
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME, heartbeat_retry_count=3),
            logger=mock.Mock(),
        )
        connection = mock.Mock()
        connection.peer_id = "worker:compute node@10.0.0.2:5000"
        connection.node_name = COMPUTE_NODE_NAME
        connection.peer_address = "10.0.0.2"
        connection.peer_port = 5000
        connection.sock = mock.Mock()
        connection.io_lock = threading.Lock()
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = [connection]
        runtime.registry.record_heartbeat_failure.side_effect = [1, 2, 3, 4]
        runtime.registry.remove_worker.return_value = connection
        runtime.registry.total_registered_gflops.return_value = 0.0
        runtime.registry.count_workers.return_value = 0
        runtime.registry.count_registered_hardware.return_value = 0

        runtime._send_heartbeat_once()
        runtime._send_heartbeat_once()
        runtime._send_heartbeat_once()
        runtime._send_heartbeat_once()

        self.assertEqual(send_message_mock.call_count, 16)
        self.assertEqual(runtime.registry.record_heartbeat_failure.call_count, 4)
        runtime.registry.remove_worker.assert_called_once_with(connection.peer_id)
        safe_close_mock.assert_called_once_with(connection.sock)
        runtime.logger.log.assert_called()

    @mock.patch("main_node.heartbeat.recv_message")
    @mock.patch("main_node.heartbeat.send_message")
    @mock.patch("main_node.heartbeat.build_heartbeat")
    def test_send_heartbeat_once_marks_alive_worker_on_ack(
        self,
        build_heartbeat_mock: mock.Mock,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        connection = mock.Mock()
        connection.peer_id = "worker:compute node@10.0.0.2:5000"
        connection.node_name = COMPUTE_NODE_NAME
        connection.peer_address = "10.0.0.2"
        connection.peer_port = 5000
        connection.sock = mock.Mock()
        connection.io_lock = threading.Lock()
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = [connection]
        runtime.registry.total_registered_gflops.return_value = 149.0
        runtime.registry.count_workers.return_value = 1
        runtime.registry.count_registered_hardware.return_value = 2
        build_heartbeat_mock.return_value = build_heartbeat(MAIN_NODE_NAME, unix_time_ms=123456)
        recv_message_mock.return_value = build_heartbeat_ok(COMPUTE_NODE_NAME, 123456, received_unix_time_ms=124000)

        runtime._send_heartbeat_once()

        send_message_mock.assert_called_once()
        runtime.registry.mark_heartbeat.assert_called_once_with(connection.peer_id, sent_at=124.0)
        runtime.registry.remove_worker.assert_not_called()
        runtime.logger.log.assert_not_called()

    @mock.patch("main_node.heartbeat.HeartbeatCoordinator.send_heartbeat_with_retry")
    def test_send_heartbeat_once_skips_when_no_workers_are_registered(
        self,
        send_with_retry_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = []

        runtime._send_heartbeat_once()

        runtime.registry.list_workers.assert_called_once_with()
        send_with_retry_mock.assert_not_called()

    def test_print_cluster_compute_capacity_includes_method_totals(self) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=MAIN_NODE_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.total_registered_gflops.return_value = 149.0
        runtime.registry.count_workers.return_value = 1
        runtime.registry.count_registered_hardware.return_value = 2
        runtime.registry.total_registered_gflops_by_method.return_value = {
            METHOD_GEMV: 24.0,
            METHOD_CONV2D: 125.0,
        }

        runtime._print_cluster_compute_capacity()

        logged = runtime.logger.info.call_args.args[0]
        self.assertIn("Current cluster compute capacity", logged)
        self.assertIn("gemv_effective_gflops", logged)
        self.assertIn("conv2d_effective_gflops", logged)


if __name__ == "__main__":
    unittest.main()
