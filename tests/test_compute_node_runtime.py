"""Compute-node runtime tests."""

import unittest
from unittest import mock

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, DiscoveryResult, HardwareProfile
from compute_node.runtime import ComputeNodeRuntime
from config import AppConfig
from constants import COMPUTE_NODE_NAME, MAIN_NODE_NAME
from runtime_protocol import Heartbeat, MessageKind, RegisterOk, RuntimeEnvelope


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


class ComputeNodeRuntimeTests(unittest.TestCase):
    """Validate compute-node runtime behavior after discovery succeeds."""

    @mock.patch("builtins.print")
    @mock.patch("compute_node.runtime.load_compute_performance_summary")
    @mock.patch("compute_node.runtime.collect_hardware_profile")
    def test_run_registers_and_records_heartbeat(
        self,
        collect_hardware_profile_mock: mock.Mock,
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
        performance = ComputePerformanceSummary(
            hardware_count=2,
            ranked_hardware=[
                ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=125.0, rank=1),
                ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=2),
            ],
        )
        collect_hardware_profile_mock.return_value = hardware
        load_compute_performance_summary_mock.return_value = performance
        fake_session = _FakeSession(
            messages=[
                RuntimeEnvelope(
                    kind=MessageKind.HEARTBEAT,
                    heartbeat=Heartbeat(main_node_name=MAIN_NODE_NAME, unix_time_ms=123456),
                ),
                None,
            ],
            register_ok=RegisterOk(
                main_node_name=MAIN_NODE_NAME,
                main_node_ip="10.0.0.5",
                main_node_port=52020,
            ),
        )

        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: fake_session,
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
        self.assertEqual(fake_session.register_args[2], performance)
        self.assertEqual(len(fake_session.sent_messages), 1)
        self.assertEqual(fake_session.sent_messages[0].kind, MessageKind.HEARTBEAT_OK)
        self.assertEqual(fake_session.sent_messages[0].heartbeat_ok.heartbeat_unix_time_ms, 123456)
        self.assertIsNotNone(runtime.heartbeat_state.last_heartbeat)
        assert runtime.heartbeat_state.last_heartbeat is not None
        self.assertEqual(runtime.heartbeat_state.last_heartbeat.unix_time_ms, 123456)
        self.assertTrue(print_mock.called)

    @mock.patch("compute_node.runtime.load_compute_performance_summary")
    @mock.patch("compute_node.runtime.collect_hardware_profile")
    def test_run_reports_registration_failure(
        self,
        collect_hardware_profile_mock: mock.Mock,
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
        load_compute_performance_summary_mock.return_value = ComputePerformanceSummary(
            hardware_count=1,
            ranked_hardware=[ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=1)],
        )

        class _BrokenSession(_FakeSession):
            def connect(self) -> None:
                raise OSError("connect failed")

        runtime = ComputeNodeRuntime(
            config=AppConfig(node_name=COMPUTE_NODE_NAME),
            main_node_host="10.0.0.5",
            main_node_port=52020,
            logger=mock.Mock(),
            session_factory=lambda *args, **kwargs: _BrokenSession([], RegisterOk("", "", 0)),
        )

        result = runtime.run()

        self.assertFalse(result.success)
        self.assertIn("Unable to join main-node TCP runtime", result.message)


if __name__ == "__main__":
    unittest.main()
