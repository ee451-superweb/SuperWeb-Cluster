"""Main-node registry tests."""

import unittest
from unittest import mock

from common.types import HardwareProfile
from main_node.registry import HomeClusterRegistry


class MainNodeRegistryTests(unittest.TestCase):
    """Validate worker/client pool separation."""

    def test_registers_workers_and_clients_in_separate_pools(self) -> None:
        registry = HomeClusterRegistry()
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

        worker = registry.register_worker("home computer", "10.0.0.2", 5000, hardware, mock.Mock())
        client = registry.register_client("home client", "10.0.0.3", 6000, mock.Mock())

        self.assertEqual(registry.count_workers(), 1)
        self.assertEqual(registry.count_clients(), 1)
        self.assertEqual(registry.list_workers()[0].peer_id, worker.peer_id)
        self.assertEqual(registry.list_clients()[0].peer_id, client.peer_id)
        self.assertEqual(registry.count(), 2)


if __name__ == "__main__":
    unittest.main()