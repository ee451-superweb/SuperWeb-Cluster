"""Minimal multicast helper tests."""

import unittest
from unittest import mock

from adapters import network
from core.config import AppConfig
from core.constants import (
    COMPUTE_NODE_NAME,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_MULTICAST_TTL,
)
from tests.support import require_integration
from discovery.multicast import (
    close,
    create_receiver,
    create_sender,
    describe_packet,
    recv_discover,
    recv_packet,
    send_discover,
)
from wire.discovery_protocol import build_discover_message


class MulticastTests(unittest.TestCase):
    """Validate basic sender socket creation."""

    @mock.patch("discovery.multicast.network.set_socket_timeout")
    @mock.patch("discovery.multicast.network.configure_multicast_sender")
    @mock.patch("discovery.multicast.network.resolve_multicast_interface_ip")
    @mock.patch("discovery.multicast.network.create_udp_socket")
    def test_create_sender_returns_closable_socket(
        self,
        create_udp_socket_mock: mock.Mock,
        resolve_multicast_interface_ip_mock: mock.Mock,
        configure_multicast_sender_mock: mock.Mock,
        set_socket_timeout_mock: mock.Mock,
    ) -> None:
        fake_socket = mock.Mock()
        create_udp_socket_mock.return_value = fake_socket
        resolve_multicast_interface_ip_mock.return_value = "192.168.1.10"

        endpoint = create_sender(AppConfig())
        self.assertIs(endpoint.sock, fake_socket)
        fake_socket.bind.assert_called_once()
        configure_multicast_sender_mock.assert_called_once_with(
            fake_socket,
            DEFAULT_MULTICAST_TTL,
            interface_ip="192.168.1.10",
        )
        set_socket_timeout_mock.assert_called_once()
        close(endpoint)

    @mock.patch("discovery.multicast.network.set_socket_timeout")
    @mock.patch("discovery.multicast.network.configure_multicast_receiver")
    @mock.patch("discovery.multicast.network.resolve_multicast_interface_ip")
    @mock.patch("discovery.multicast.network.create_udp_socket")
    @mock.patch("discovery.multicast.sys.platform", "darwin")
    def test_create_receiver_joins_selected_interface(
        self,
        create_udp_socket_mock: mock.Mock,
        resolve_multicast_interface_ip_mock: mock.Mock,
        configure_multicast_receiver_mock: mock.Mock,
        set_socket_timeout_mock: mock.Mock,
    ) -> None:
        fake_socket = mock.Mock()
        create_udp_socket_mock.return_value = fake_socket
        resolve_multicast_interface_ip_mock.return_value = "192.168.1.10"
        configure_multicast_receiver_mock.return_value = b"membership"

        endpoint = create_receiver(AppConfig())

        self.assertIs(endpoint.sock, fake_socket)
        self.assertEqual(endpoint.membership, b"membership")
        create_udp_socket_mock.assert_called_once_with(reuse_port=True)
        configure_multicast_receiver_mock.assert_called_once_with(
            fake_socket,
            group=DEFAULT_MULTICAST_GROUP,
            port=DEFAULT_DISCOVERY_PORT,
            interface_ip="192.168.1.10",
        )
        set_socket_timeout_mock.assert_called_once()

    def test_recv_packet_returns_raw_payload(self) -> None:
        fake_socket = mock.Mock()
        payload = build_discover_message(COMPUTE_NODE_NAME)
        fake_socket.recvfrom.return_value = (payload, ("10.0.0.2", 5000))
        endpoint = mock.Mock()
        endpoint.sock = fake_socket

        packet = recv_packet(endpoint, 2048)
        self.assertEqual(packet, (("10.0.0.2", 5000), payload))

    def test_recv_discover_parses_valid_discover_message(self) -> None:
        fake_socket = mock.Mock()
        payload = build_discover_message(COMPUTE_NODE_NAME)
        fake_socket.recvfrom.return_value = (payload, ("10.0.0.2", 5000))
        fake_socket.gettimeout.return_value = 1.0
        endpoint = mock.Mock()
        endpoint.sock = fake_socket

        discovered = recv_discover(endpoint, DEFAULT_BUFFER_SIZE)
        self.assertEqual(discovered, (("10.0.0.2", 5000), payload))

    def test_send_discover_prints_payload(self) -> None:
        config = AppConfig(node_name="unit-test-node")
        endpoint = mock.Mock()
        endpoint.sock = mock.Mock()
        payload = build_discover_message(config.node_name)

        send_discover(endpoint, config, config.node_name)

        endpoint.sock.sendto.assert_called_once_with(
            payload,
            (config.multicast_group, config.udp_port),
        )

    @require_integration("Live multicast probes are reserved for integration runs.")
    def test_live_multicast_sender_can_send_probe(self) -> None:
        config = AppConfig(node_name="unit-live-probe", discovery_timeout=0.2)
        endpoint = create_sender(config)
        try:
            local_host, local_port = endpoint.sock.getsockname()
            send_discover(endpoint, config, config.node_name)

            self.assertIsInstance(local_host, str)
            self.assertGreater(local_port, 0)
        finally:
            close(endpoint)

    @require_integration("Live multicast loopback checks are reserved for integration runs.")
    def test_live_multicast_loopback_chain(self) -> None:
        config = AppConfig(node_name="unit-loopback-probe", discovery_timeout=1.0)
        payload = build_discover_message(config.node_name)

        receiver = create_receiver(config)
        sender = create_sender(config)
        try:
            send_discover(sender, config, config.node_name)
            packet = recv_discover(receiver, config.buffer_size)

            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet[1], payload)
            self.assertIn("mDNS PTR query", describe_packet(packet[1]))
        finally:
            close(sender)
            close(receiver)


if __name__ == "__main__":
    unittest.main()


