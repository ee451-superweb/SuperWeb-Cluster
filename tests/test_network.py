"""Network helper tests."""

import socket
import struct
import unittest
from unittest import mock

from adapters import network
from app.constants import DEFAULT_DISCOVERY_PORT, DEFAULT_MULTICAST_GROUP, DEFAULT_MULTICAST_TTL


class NetworkTests(unittest.TestCase):
    """Validate multicast-specific socket configuration."""

    def test_create_udp_socket_sets_requested_reuse_flags(self) -> None:
        fake_socket = mock.Mock()

        with mock.patch("adapters.network.socket.socket", return_value=fake_socket):
            sock = network.create_udp_socket(reuse_port=True)

        self.assertIs(sock, fake_socket)
        fake_socket.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            fake_socket.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    def test_print_actual_multicast_interface_ip(self) -> None:
        primary_ip = network.resolve_local_ip()
        interface_ip = network.resolve_multicast_interface_ip(DEFAULT_MULTICAST_GROUP, DEFAULT_DISCOVERY_PORT)

        print(f"[LIVE] primary_local_ip={primary_ip}", flush=True)
        print(f"[LIVE] multicast_interface_ip={interface_ip or '(default)'}", flush=True)

        self.assertIsInstance(interface_ip, str)

    def test_configure_multicast_sender_sets_ttl_and_interface(self) -> None:
        fake_socket = mock.Mock()

        network.configure_multicast_sender(
            fake_socket,
            ttl=DEFAULT_MULTICAST_TTL,
            interface_ip="192.168.1.10",
        )

        self.assertEqual(fake_socket.setsockopt.call_count, 2)
        fake_socket.setsockopt.assert_any_call(
            socket.IPPROTO_IP,
            socket.IP_MULTICAST_TTL,
            struct.pack("B", DEFAULT_MULTICAST_TTL),
        )
        fake_socket.setsockopt.assert_any_call(
            socket.IPPROTO_IP,
            socket.IP_MULTICAST_IF,
            socket.inet_aton("192.168.1.10"),
        )

    def test_configure_multicast_receiver_uses_selected_interface(self) -> None:
        fake_socket = mock.Mock()

        membership = network.configure_multicast_receiver(
            fake_socket,
            group=DEFAULT_MULTICAST_GROUP,
            port=DEFAULT_DISCOVERY_PORT,
            interface_ip="192.168.1.10",
        )

        fake_socket.bind.assert_called_once_with(("", DEFAULT_DISCOVERY_PORT))
        fake_socket.setsockopt.assert_called_once_with(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            membership,
        )
        self.assertEqual(
            membership,
            struct.pack(
                "4s4s",
                socket.inet_aton(DEFAULT_MULTICAST_GROUP),
                socket.inet_aton("192.168.1.10"),
            ),
        )

    @mock.patch("adapters.network.resolve_local_ip")
    def test_resolve_multicast_interface_ip_prefers_primary_route(
        self,
        resolve_local_ip_mock: mock.Mock,
    ) -> None:
        resolve_local_ip_mock.return_value = "192.168.1.148"

        interface_ip = network.resolve_multicast_interface_ip(DEFAULT_MULTICAST_GROUP, DEFAULT_DISCOVERY_PORT)

        self.assertEqual(interface_ip, "192.168.1.148")
        resolve_local_ip_mock.assert_called_once_with()

    @mock.patch("adapters.network.resolve_local_ip")
    def test_resolve_multicast_interface_ip_falls_back_from_loopback(
        self,
        resolve_local_ip_mock: mock.Mock,
    ) -> None:
        resolve_local_ip_mock.side_effect = ["127.0.0.1", "192.168.1.10"]

        interface_ip = network.resolve_multicast_interface_ip(DEFAULT_MULTICAST_GROUP, DEFAULT_DISCOVERY_PORT)

        self.assertEqual(interface_ip, "192.168.1.10")
        self.assertEqual(resolve_local_ip_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()

