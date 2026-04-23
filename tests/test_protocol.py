"""Protocol format tests."""

import unittest

from core.constants import (
    COMPUTE_NODE_NAME,
    DEFAULT_TCP_PORT,
    MAIN_NODE_NAME,
    MDNS_SERVICE_TYPE,
)
from wire.discovery_protocol import (
    describe_discovery_message,
    build_announce_message,
    build_discover_message,
    normalize_manual_address,
    parse_announce_message,
    parse_discover_message,
)


class ProtocolTests(unittest.TestCase):
    """Validate superweb-cluster mDNS discovery message formatting."""

    def test_discover_round_trip(self) -> None:
        message = build_discover_message(COMPUTE_NODE_NAME)
        self.assertTrue(parse_discover_message(message))
        self.assertEqual(
            describe_discovery_message(message),
            f"mDNS PTR query for {MDNS_SERVICE_TYPE}",
        )

    def test_announce_round_trip(self) -> None:
        message = build_announce_message("10.0.0.5", DEFAULT_TCP_PORT, MAIN_NODE_NAME)
        payload = parse_announce_message(message)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.host, "10.0.0.5")
        self.assertEqual(payload.port, DEFAULT_TCP_PORT)
        self.assertEqual(payload.node_name, MAIN_NODE_NAME)
        self.assertIn(MAIN_NODE_NAME, describe_discovery_message(message))

    def test_manual_address_defaults_port(self) -> None:
        host, port = normalize_manual_address("example.local", DEFAULT_TCP_PORT)
        self.assertEqual(host, "example.local")
        self.assertEqual(port, DEFAULT_TCP_PORT)


if __name__ == "__main__":
    unittest.main()


