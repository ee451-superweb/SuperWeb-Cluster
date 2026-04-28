"""Runtime protobuf protocol tests."""

import socket
import unittest
from unittest import mock

from core.float32_codec import pack_float32_values, unpack_float32_bytes
from core.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile
from core.constants import (
    COMPUTE_NODE_NAME,
    CONV2D_CLIENT_RESPONSE_STATS_ONLY,
    MAIN_NODE_NAME,
    METHOD_GEMM,
    METHOD_GEMV,
    METHOD_CONV2D,
    STATUS_ACCEPTED,
    STATUS_OK,
    SUPERWEB_CLIENT_NAME,
)
from wire.internal_protocol.control_plane_codec import encode_envelope, parse_envelope
from wire.internal_protocol.transport import (
    GemmRequestPayload,
    GemmResponsePayload,
    GemmResultPayload,
    GemmTaskPayload,
    MessageKind,
    Conv2dRequestPayload,
    Conv2dResponsePayload,
    Conv2dResultPayload,
    Conv2dTaskPayload,
    TransferMode,
    build_client_join,
    build_client_request,
    build_client_response,
    build_heartbeat,
    build_heartbeat_ok,
    build_register_ok,
    build_register_worker,
    build_task_accept,
    build_task_assign,
    build_task_fail,
    build_task_result,
    recv_message,
    send_message,
)


class RuntimeProtocolTests(unittest.TestCase):
    """Validate framed protobuf runtime messages."""

    def test_register_worker_round_trip(self) -> None:
        hardware = HardwareProfile(
            hostname="node-a",
            local_ip="10.0.0.2",
            mac_address="aa:bb:cc:dd:ee:ff",
            system="Windows",
            release="11",
            machine="AMD64",
            processor="x86_64",
            logical_cpu_count=16,
            memory_bytes=32 * 1024 * 1024 * 1024,
        )
        performance = ComputePerformanceSummary(
            hardware_count=2,
            ranked_hardware=[
                ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=122.992902574, rank=1),
                ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=18.733914864, rank=2),
            ],
        )

        encoded = encode_envelope(build_register_worker(COMPUTE_NODE_NAME, hardware, performance))
        decoded = parse_envelope(encoded)

        self.assertEqual(decoded.kind, MessageKind.REGISTER_WORKER)
        self.assertIsNotNone(decoded.register_worker)
        assert decoded.register_worker is not None
        self.assertEqual(decoded.register_worker.node_name, COMPUTE_NODE_NAME)
        self.assertEqual(decoded.register_worker.hardware.hostname, "node-a")
        self.assertEqual(decoded.register_worker.hardware.logical_cpu_count, 16)
        self.assertEqual(decoded.register_worker.performance.hardware_count, 2)
        self.assertEqual(decoded.register_worker.performance.ranked_hardware[0].hardware_type, "cuda")
        self.assertAlmostEqual(decoded.register_worker.performance.ranked_hardware[0].effective_gflops, 122.992902574)

    def test_register_ok_round_trip(self) -> None:
        encoded = encode_envelope(build_register_ok("10.0.0.5", 52020, node_id="worker-1", main_node_name=MAIN_NODE_NAME))
        decoded = parse_envelope(encoded)

        self.assertEqual(decoded.kind, MessageKind.REGISTER_OK)
        self.assertIsNotNone(decoded.register_ok)
        assert decoded.register_ok is not None
        self.assertEqual(decoded.register_ok.main_node_ip, "10.0.0.5")
        self.assertEqual(decoded.register_ok.main_node_port, 52020)
        self.assertEqual(decoded.register_ok.node_id, "worker-1")

    def test_heartbeat_ok_round_trip(self) -> None:
        encoded = encode_envelope(
            build_heartbeat_ok(
                COMPUTE_NODE_NAME,
                123456,
                123999,
                active_task_ids=["task-1", "task-2"],
            )
        )
        decoded = parse_envelope(encoded)

        self.assertEqual(decoded.kind, MessageKind.HEARTBEAT_OK)
        self.assertIsNotNone(decoded.heartbeat_ok)
        assert decoded.heartbeat_ok is not None
        self.assertEqual(decoded.heartbeat_ok.node_name, COMPUTE_NODE_NAME)
        self.assertEqual(decoded.heartbeat_ok.heartbeat_unix_time_ms, 123456)
        self.assertEqual(decoded.heartbeat_ok.received_unix_time_ms, 123999)
        self.assertEqual(decoded.heartbeat_ok.active_task_ids, ("task-1", "task-2"))

    def test_client_join_request_response_round_trip(self) -> None:
        vector_data = pack_float32_values([1.0, 2.0, 3.0, 4.0])
        output_vector = pack_float32_values([5.0, 6.0])
        join = parse_envelope(encode_envelope(build_client_join(SUPERWEB_CLIENT_NAME)))
        request = parse_envelope(
            encode_envelope(
                build_client_request(
                    SUPERWEB_CLIENT_NAME,
                    "req-1",
                    METHOD_GEMV,
                    vector_data,
                    object_id="input_matrix/default",
                    stream_id="stream-1",
                    iteration_count=7,
                )
            )
        )
        response = parse_envelope(
            encode_envelope(
                build_client_response(
                    request_id="req-1",
                    status_code=STATUS_OK,
                    method=METHOD_GEMV,
                    object_id="input_matrix/default",
                    stream_id="stream-1",
                    output_vector=output_vector,
                    worker_count=2,
                    client_count=1,
                    client_id="client-1",
                    iteration_count=7,
                )
            )
        )

        self.assertEqual(join.kind, MessageKind.CLIENT_JOIN)
        self.assertEqual(join.client_join.client_name, SUPERWEB_CLIENT_NAME)
        self.assertEqual(request.kind, MessageKind.CLIENT_REQUEST)
        self.assertEqual(request.client_request.method, METHOD_GEMV)
        self.assertEqual(request.client_request.vector_length, 4)
        self.assertEqual(request.client_request.iteration_count, 7)
        self.assertEqual(unpack_float32_bytes(request.client_request.vector_data), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.status_code, STATUS_OK)
        self.assertEqual(response.client_response.worker_count, 2)
        self.assertEqual(response.client_response.client_count, 1)
        self.assertEqual(response.client_response.client_id, "client-1")
        self.assertEqual(response.client_response.iteration_count, 7)
        self.assertEqual(unpack_float32_bytes(response.client_response.output_vector), [5.0, 6.0])

    def test_task_messages_round_trip(self) -> None:
        vector_data = pack_float32_values([1.0, 2.0, 3.0, 4.0])
        result_vector = pack_float32_values([9.0, 10.0])
        assign = parse_envelope(
            encode_envelope(
                build_task_assign(
                    request_id="req-1",
                    node_id=COMPUTE_NODE_NAME,
                    task_id="req-1:worker-a",
                    method=METHOD_GEMV,
                    row_start=10,
                    row_end=12,
                    vector_data=vector_data,
                    object_id="input_matrix/default",
                    stream_id="stream-1",
                    iteration_count=11,
                )
            )
        )
        accept = parse_envelope(
            encode_envelope(build_task_accept("req-1", COMPUTE_NODE_NAME, "req-1:worker-a", STATUS_ACCEPTED))
        )
        fail = parse_envelope(
            encode_envelope(build_task_fail("req-1", COMPUTE_NODE_NAME, "req-1:worker-a", 500, "boom"))
        )
        result = parse_envelope(
            encode_envelope(
                build_task_result(
                    "req-1",
                    COMPUTE_NODE_NAME,
                    "req-1:worker-a",
                    STATUS_OK,
                    row_start=10,
                    row_end=12,
                    output_vector=result_vector,
                    iteration_count=11,
                )
            )
        )

        self.assertEqual(assign.kind, MessageKind.TASK_ASSIGN)
        self.assertEqual(assign.task_assign.row_start, 10)
        self.assertEqual(assign.task_assign.row_end, 12)
        self.assertEqual(assign.task_assign.iteration_count, 11)
        self.assertEqual(assign.task_assign.transfer_mode, TransferMode.UNSPECIFIED)
        self.assertEqual(unpack_float32_bytes(assign.task_assign.vector_data), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(accept.kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(accept.task_accept.status_code, STATUS_ACCEPTED)
        self.assertEqual(fail.kind, MessageKind.TASK_FAIL)
        self.assertEqual(fail.task_fail.error_message, "boom")
        self.assertEqual(result.kind, MessageKind.TASK_RESULT)
        self.assertEqual(result.task_result.output_length, 2)
        self.assertEqual(result.task_result.iteration_count, 11)
        self.assertEqual(unpack_float32_bytes(result.task_result.output_vector), [9.0, 10.0])

    def test_gemm_messages_use_typed_payloads(self) -> None:
        output_vector = pack_float32_values([0.25] * (512 * 1024))
        request = parse_envelope(
            encode_envelope(
                build_client_request(
                    SUPERWEB_CLIENT_NAME,
                    "req-gemm",
                    METHOD_GEMM,
                    b"",
                    size="small",
                    object_id="gemm/small",
                    stream_id="stream-gemm",
                    iteration_count=2,
                    request_payload=GemmRequestPayload(),
                )
            )
        )
        assign = parse_envelope(
            encode_envelope(
                build_task_assign(
                    request_id="req-gemm",
                    node_id=COMPUTE_NODE_NAME,
                    task_id="req-gemm:worker-a",
                    method=METHOD_GEMM,
                    size="small",
                    row_start=0,
                    row_end=0,
                    vector_data=b"",
                    object_id="gemm/small",
                    stream_id="stream-gemm",
                    iteration_count=2,
                    task_payload=GemmTaskPayload(
                        m_start=0, m_end=512, m=1024, n=1024, k=1024,
                    ),
                )
            )
        )
        result = parse_envelope(
            encode_envelope(
                build_task_result(
                    "req-gemm",
                    COMPUTE_NODE_NAME,
                    "req-gemm:worker-a",
                    STATUS_OK,
                    iteration_count=2,
                    result_payload=GemmResultPayload(
                        m_start=0,
                        m_end=512,
                        output_length=512 * 1024,
                        output_vector=output_vector,
                    ),
                )
            )
        )
        response = parse_envelope(
            encode_envelope(
                build_client_response(
                    "req-gemm",
                    STATUS_OK,
                    method=METHOD_GEMM,
                    size="small",
                    object_id="gemm/small",
                    stream_id="stream-gemm",
                    iteration_count=2,
                    task_id="req-gemm",
                    response_payload=GemmResponsePayload(
                        output_length=1024 * 1024,
                        output_vector=output_vector,
                    ),
                )
            )
        )

        self.assertIsInstance(request.client_request.request_payload, GemmRequestPayload)
        self.assertEqual(request.client_request.method, METHOD_GEMM)
        self.assertEqual(request.client_request.size, "small")
        self.assertIsInstance(assign.task_assign.task_payload, GemmTaskPayload)
        self.assertEqual(assign.task_assign.m_start, 0)
        self.assertEqual(assign.task_assign.m_end, 512)
        self.assertEqual(assign.task_assign.m, 1024)
        self.assertEqual(assign.task_assign.n, 1024)
        self.assertEqual(assign.task_assign.k, 1024)
        self.assertIsInstance(result.task_result.result_payload, GemmResultPayload)
        self.assertEqual(result.task_result.m_start, 0)
        self.assertEqual(result.task_result.m_end, 512)
        self.assertEqual(result.task_result.output_length, 512 * 1024)
        self.assertEqual(len(result.task_result.output_vector), 512 * 1024 * 4)
        self.assertIsInstance(response.client_response.response_payload, GemmResponsePayload)
        self.assertEqual(response.client_response.output_length, 1024 * 1024)

    def test_conv2d_messages_use_typed_payloads(self) -> None:
        weight_data = pack_float32_values([0.5] * (3 * 3 * 4 * 2))
        output_vector = pack_float32_values([1.0] * (8 * 8 * 2))
        request = parse_envelope(
            encode_envelope(
                build_client_request(
                    SUPERWEB_CLIENT_NAME,
                    "req-spatial",
                    METHOD_CONV2D,
                    b"",
                    size="small",
                    object_id="conv2d/small",
                    stream_id="stream-spatial",
                    tensor_h=8,
                    tensor_w=8,
                    channels_in=4,
                    channels_out=8,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
        )
        assign = parse_envelope(
            encode_envelope(
                build_task_assign(
                    request_id="req-spatial",
                    node_id=COMPUTE_NODE_NAME,
                    task_id="req-spatial:worker-a:0",
                    method=METHOD_CONV2D,
                    size="small",
                    row_start=0,
                    row_end=0,
                    vector_data=b"",
                    object_id="conv2d/small",
                    stream_id="stream-spatial",
                    iteration_count=3,
                    transfer_mode=TransferMode.ARTIFACT_PREFERRED,
                    start_oc=2,
                    end_oc=4,
                    tensor_h=8,
                    tensor_w=8,
                    channels_in=4,
                    channels_out=8,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    weight_data=weight_data,
                )
            )
        )
        result = parse_envelope(
            encode_envelope(
                build_task_result(
                    "req-spatial",
                    COMPUTE_NODE_NAME,
                    "req-spatial:worker-a:0",
                    STATUS_OK,
                    row_start=0,
                    row_end=0,
                    output_vector=output_vector,
                    output_length=8 * 8 * 2,
                    iteration_count=3,
                    start_oc=2,
                    end_oc=4,
                    output_h=8,
                    output_w=8,
                    result_artifact_id="",
                )
            )
        )

        self.assertIsInstance(request.client_request.request_payload, Conv2dRequestPayload)
        self.assertEqual(request.client_request.tensor_h, 8)
        self.assertEqual(request.client_request.channels_out, 8)
        self.assertEqual(request.client_request.vector_length, 0)
        self.assertIsInstance(assign.task_assign.task_payload, Conv2dTaskPayload)
        self.assertEqual(assign.task_assign.transfer_mode, TransferMode.ARTIFACT_PREFERRED)
        self.assertEqual(assign.task_assign.start_oc, 2)
        self.assertEqual(assign.task_assign.end_oc, 4)
        self.assertEqual(assign.task_assign.channels_in, 4)
        self.assertEqual(assign.task_assign.weight_data, weight_data)
        self.assertIsInstance(result.task_result.result_payload, Conv2dResultPayload)
        self.assertEqual(result.task_result.output_h, 8)
        self.assertEqual(result.task_result.output_w, 8)
        self.assertEqual(result.task_result.output_length, 8 * 8 * 2)
        self.assertEqual(result.task_result.start_oc, 2)

    def test_conv2d_stats_only_request_and_response_round_trip(self) -> None:
        request = parse_envelope(
            encode_envelope(
                build_client_request(
                    SUPERWEB_CLIENT_NAME,
                    "req-stats",
                    METHOD_CONV2D,
                    b"",
                    size="small",
                    object_id="conv2d/small",
                    stream_id="stream-stats",
                    tensor_h=8,
                    tensor_w=8,
                    channels_in=4,
                    channels_out=8,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    conv2d_client_response_mode=CONV2D_CLIENT_RESPONSE_STATS_ONLY,
                    conv2d_stats_max_samples=16,
                )
            )
        )
        self.assertIsInstance(request.client_request.request_payload, Conv2dRequestPayload)
        assert request.client_request.conv2d_payload is not None
        self.assertEqual(
            request.client_request.conv2d_payload.client_response_mode,
            CONV2D_CLIENT_RESPONSE_STATS_ONLY,
        )
        self.assertEqual(request.client_request.conv2d_payload.stats_max_samples, 16)

        task_assign = parse_envelope(
            encode_envelope(
                build_task_assign(
                    request_id="req-stats",
                    node_id=COMPUTE_NODE_NAME,
                    task_id="req-stats:worker-a:0",
                    method=METHOD_CONV2D,
                    size="small",
                    object_id="conv2d/small",
                    stream_id="stream-stats",
                    iteration_count=1,
                    transfer_mode=TransferMode.INLINE_PREFERRED,
                    task_payload=Conv2dTaskPayload(
                        start_oc=0,
                        end_oc=4,
                        tensor_h=8,
                        tensor_w=8,
                        channels_in=4,
                        channels_out=8,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        weight_data=b"\x00" * (3 * 3 * 4 * 4 * 4),
                        client_response_mode=CONV2D_CLIENT_RESPONSE_STATS_ONLY,
                        stats_max_samples=16,
                    ),
                )
            )
        )
        task_conv2d = task_assign.task_assign.conv2d_payload
        assert task_conv2d is not None
        self.assertEqual(task_conv2d.client_response_mode, CONV2D_CLIENT_RESPONSE_STATS_ONLY)
        self.assertEqual(task_conv2d.stats_max_samples, 16)

        worker_result = parse_envelope(
            encode_envelope(
                build_task_result(
                    "req-stats",
                    COMPUTE_NODE_NAME,
                    "req-stats:worker-a:0",
                    STATUS_OK,
                    iteration_count=1,
                    result_payload=Conv2dResultPayload(
                        start_oc=0,
                        end_oc=4,
                        output_h=8,
                        output_w=8,
                        output_length=8 * 8 * 4,
                        output_vector=b"",
                        result_artifact_id="",
                        stats_element_count=8 * 8 * 4,
                        stats_sum=12.5,
                        stats_sum_squares=75.25,
                        stats_samples=(0.5, 1.0, 1.5),
                    ),
                )
            )
        )
        result_conv2d = worker_result.task_result.conv2d_payload
        assert result_conv2d is not None
        self.assertEqual(result_conv2d.stats_element_count, 8 * 8 * 4)
        self.assertAlmostEqual(result_conv2d.stats_sum, 12.5)
        self.assertAlmostEqual(result_conv2d.stats_sum_squares, 75.25)
        self.assertEqual(result_conv2d.stats_samples, (0.5, 1.0, 1.5))

        stats_payload = Conv2dResponsePayload(
            output_length=64,
            output_vector=b"",
            result_artifact_id="",
            stats_element_count=64,
            stats_sum=10.0,
            stats_sum_squares=100.0,
            stats_samples=(1.0, 2.0, 3.0),
        )
        response = parse_envelope(
            encode_envelope(
                build_client_response(
                    request_id="req-stats",
                    status_code=STATUS_OK,
                    method=METHOD_CONV2D,
                    object_id="conv2d/small",
                    stream_id="stream-stats",
                    worker_count=1,
                    client_count=1,
                    client_id="client-1",
                    iteration_count=1,
                    response_payload=stats_payload,
                )
            )
        )
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        cp = response.client_response.conv2d_payload
        assert cp is not None
        self.assertEqual(cp.output_length, 64)
        self.assertEqual(cp.stats_element_count, 64)
        self.assertAlmostEqual(cp.stats_sum, 10.0)
        self.assertAlmostEqual(cp.stats_sum_squares, 100.0)
        self.assertEqual(cp.stats_samples, (1.0, 2.0, 3.0))
        self.assertEqual(response.client_response.result_artifact_id, "")

    def test_framed_send_receive_round_trip(self) -> None:
        left, right = socket.socketpair()
        try:
            left.settimeout(1.0)
            right.settimeout(1.0)

            send_message(left, build_heartbeat(MAIN_NODE_NAME, 123456))
            received = recv_message(right, max_size=4096)

            self.assertIsNotNone(received)
            assert received is not None
            self.assertEqual(received.kind, MessageKind.HEARTBEAT)
            self.assertIsNotNone(received.heartbeat)
            assert received.heartbeat is not None
            self.assertEqual(received.heartbeat.unix_time_ms, 123456)
        finally:
            left.close()
            right.close()

    def test_recv_message_rejects_non_bytes_socket_chunks(self) -> None:
        sock = mock.Mock()
        sock.recv.return_value = mock.Mock()

        with self.assertRaisesRegex(TypeError, "socket\\.recv\\(\\) must return bytes-like data"):
            recv_message(sock, max_size=4096)


if __name__ == "__main__":
    unittest.main()


