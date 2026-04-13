"""Runtime protobuf protocol tests."""

import socket
import unittest

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile
from constants import COMPUTE_NODE_NAME, MAIN_NODE_NAME, SUPERWEB_CLIENT_NAME
from runtime_protocol import (
    MessageKind,
    build_client_join,
    build_client_request,
    build_client_response,
    build_heartbeat,
    build_heartbeat_ok,
    build_register_ok,
    build_register_worker,
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
        encoded = encode_envelope(build_register_ok("10.0.0.5", 52020, MAIN_NODE_NAME))
        decoded = parse_envelope(encoded)

        self.assertEqual(decoded.kind, MessageKind.REGISTER_OK)
        self.assertIsNotNone(decoded.register_ok)
        assert decoded.register_ok is not None
        self.assertEqual(decoded.register_ok.main_node_ip, "10.0.0.5")
        self.assertEqual(decoded.register_ok.main_node_port, 52020)

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
        join = parse_envelope(encode_envelope(build_client_join(SUPERWEB_CLIENT_NAME)))
        request = parse_envelope(encode_envelope(build_client_request(SUPERWEB_CLIENT_NAME, "req-1", "cluster_status", "")))
        response = parse_envelope(
            encode_envelope(
                build_client_response(
                    request_id="req-1",
                    ok=True,
                    message="Cluster status snapshot.",
                    payload="workers=2 clients=1",
                    worker_count=2,
                    client_count=1,
                )
            )
        )

        self.assertEqual(join.kind, MessageKind.CLIENT_JOIN)
        self.assertEqual(join.client_join.client_name, SUPERWEB_CLIENT_NAME)
        self.assertEqual(request.kind, MessageKind.CLIENT_REQUEST)
        self.assertEqual(request.client_request.command, "cluster_status")
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertTrue(response.client_response.ok)
        self.assertEqual(response.client_response.worker_count, 2)
        self.assertEqual(response.client_response.client_count, 1)

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
