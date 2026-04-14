"""Runtime protobuf protocol tests."""

import socket
import unittest

from common.float32_codec import pack_float32_values, unpack_float32_bytes
from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile
from app.constants import (
    COMPUTE_NODE_NAME,
    MAIN_NODE_NAME,
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
    STATUS_ACCEPTED,
    STATUS_OK,
    SUPERWEB_CLIENT_NAME,
)
from wire.runtime import (
    MessageKind,
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
    encode_envelope,
    parse_envelope,
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
        encoded = encode_envelope(build_heartbeat_ok(COMPUTE_NODE_NAME, 123456, 123999))
        decoded = parse_envelope(encoded)

        self.assertEqual(decoded.kind, MessageKind.HEARTBEAT_OK)
        self.assertIsNotNone(decoded.heartbeat_ok)
        assert decoded.heartbeat_ok is not None
        self.assertEqual(decoded.heartbeat_ok.node_name, COMPUTE_NODE_NAME)
        self.assertEqual(decoded.heartbeat_ok.heartbeat_unix_time_ms, 123456)
        self.assertEqual(decoded.heartbeat_ok.received_unix_time_ms, 123999)

    def test_client_join_request_response_round_trip(self) -> None:
        vector_data = pack_float32_values([1.0, 2.0, 3.0, 4.0])
        output_vector = pack_float32_values([5.0, 6.0])
        join = parse_envelope(encode_envelope(build_client_join(SUPERWEB_CLIENT_NAME)))
        request = parse_envelope(
            encode_envelope(
                build_client_request(
                    SUPERWEB_CLIENT_NAME,
                    "req-1",
                    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
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
                    method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
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
        self.assertEqual(request.client_request.method, METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION)
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
                    method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
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
        self.assertEqual(unpack_float32_bytes(assign.task_assign.vector_data), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(accept.kind, MessageKind.TASK_ACCEPT)
        self.assertEqual(accept.task_accept.status_code, STATUS_ACCEPTED)
        self.assertEqual(fail.kind, MessageKind.TASK_FAIL)
        self.assertEqual(fail.task_fail.error_message, "boom")
        self.assertEqual(result.kind, MessageKind.TASK_RESULT)
        self.assertEqual(result.task_result.output_length, 2)
        self.assertEqual(result.task_result.iteration_count, 11)
        self.assertEqual(unpack_float32_bytes(result.task_result.output_vector), [9.0, 10.0])

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


if __name__ == "__main__":
    unittest.main()


