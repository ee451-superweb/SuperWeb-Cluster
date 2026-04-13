"""Main-node registry tests."""

import unittest
from unittest import mock

from common.types import ComputeHardwarePerformance, ComputePerformanceSummary, HardwareProfile
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


if __name__ == "__main__":
    unittest.main()
