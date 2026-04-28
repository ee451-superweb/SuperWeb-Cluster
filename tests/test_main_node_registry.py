"""Main-node registry tests."""

import unittest
from unittest import mock

from core.constants import METHOD_GEMV, METHOD_CONV2D
from core.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile
from core.types import MethodPerformanceSummary
from main_node.registry import ClusterRegistry


class MainNodeRegistryTests(unittest.TestCase):
    """Validate worker/client pool separation."""

    def test_registers_workers_and_clients_in_separate_pools(self) -> None:
        registry = ClusterRegistry()
        hardware = HardwareProfile(
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

        worker = registry.register_worker("compute node", "10.0.0.2", 5000, hardware, performance, mock.Mock())
        client = registry.register_client("superweb client", "10.0.0.3", 6000, mock.Mock())
        hardware_entries = registry.list_worker_hardware()

        self.assertEqual(registry.count_workers(), 1)
        self.assertEqual(registry.count_clients(), 1)
        self.assertEqual(registry.list_workers()[0].peer_id, worker.peer_id)
        self.assertEqual(registry.list_clients()[0].peer_id, client.peer_id)
        self.assertEqual(registry.count(), 2)
        self.assertEqual(registry.count_registered_hardware(), 2)
        self.assertAlmostEqual(registry.total_registered_gflops(), 149.0)
        self.assertEqual(hardware_entries[0].worker_peer_id, worker.peer_id)
        self.assertTrue(hardware_entries[0].hardware_id.startswith("hardware:"))

    def test_total_registered_gflops_by_method_reports_method_breakdown(self) -> None:
        registry = ClusterRegistry()
        hardware = HardwareProfile(
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
            method_summaries=[
                MethodPerformanceSummary(
                    method=METHOD_GEMV,
                    hardware_count=1,
                    ranked_hardware=[
                        ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=1),
                    ],
                ),
                MethodPerformanceSummary(
                    method=METHOD_CONV2D,
                    hardware_count=1,
                    ranked_hardware=[
                        ComputeHardwarePerformance(hardware_type="cuda", effective_gflops=125.0, rank=1),
                    ],
                ),
            ]
        )

        registry.register_worker("compute node", "10.0.0.2", 5000, hardware, performance, mock.Mock())

        totals = registry.total_registered_gflops_by_method()

        self.assertAlmostEqual(totals[METHOD_GEMV], 24.0)
        self.assertAlmostEqual(totals[METHOD_CONV2D], 125.0)

    def test_heartbeat_failure_counter_resets_after_success(self) -> None:
        registry = ClusterRegistry()
        hardware = HardwareProfile(
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
            ranked_hardware=[
                ComputeHardwarePerformance(hardware_type="cpu", effective_gflops=24.0, rank=1),
            ],
        )

        worker = registry.register_worker("compute node", "10.0.0.2", 5000, hardware, performance, mock.Mock())

        self.assertEqual(registry.record_heartbeat_failure(worker.peer_id), 1)
        self.assertEqual(registry.record_heartbeat_failure(worker.peer_id), 2)
        self.assertEqual(registry.get_heartbeat_failure_count(worker.peer_id), 2)

        registry.mark_heartbeat(worker.peer_id, sent_at=123.0)

        self.assertEqual(registry.get_heartbeat_failure_count(worker.peer_id), 0)


if __name__ == "__main__":
    unittest.main()
