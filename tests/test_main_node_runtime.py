"""Main node runtime tests."""

import unittest
from unittest import mock

from common.types import DiscoveryResult, HardwareProfile
from config import AppConfig
from constants import DEFAULT_TCP_PORT, HOME_CLIENT_NAME, HOME_COMPUTER_NAME, HOME_SCHEDULER_NAME
from main_node.runtime import MainNodeRuntime
from protocol import build_discover_message
from runtime_protocol import (
    MessageKind,
    build_client_join,
    build_client_request,
    build_heartbeat,
    build_heartbeat_ok,
    build_register_worker,
)


class MainNodeRuntimeTests(unittest.TestCase):
    """Validate the promoted home scheduler loop."""

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.network.safe_close")
    @mock.patch("main_node.runtime.MainNodeRuntime._close_runtime_connections")
    @mock.patch("main_node.runtime.threading.Thread")
    @mock.patch("main_node.runtime.MainNodeRuntime._create_tcp_listener")
    @mock.patch("main_node.runtime.multicast.close")
    @mock.patch("main_node.runtime.multicast.recv_packet")
    @mock.patch("main_node.runtime.network.get_local_mac_address")
    @mock.patch("main_node.runtime.network.resolve_local_ip")
    @mock.patch("main_node.runtime.network.set_socket_timeout")
    @mock.patch("main_node.runtime.multicast.create_receiver")
    def test_run_starts_loop_and_returns_success_on_shutdown(
        self,
        create_receiver_mock: mock.Mock,
        set_socket_timeout_mock: mock.Mock,
        resolve_local_ip_mock: mock.Mock,
        get_local_mac_address_mock: mock.Mock,
        recv_packet_mock: mock.Mock,
        close_mock: mock.Mock,
        create_tcp_listener_mock: mock.Mock,
        thread_mock: mock.Mock,
        close_runtime_connections_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        endpoint = mock.Mock()
        endpoint.sock = mock.Mock()
        create_receiver_mock.return_value = endpoint
        runtime_sock = mock.Mock()
        create_tcp_listener_mock.return_value = runtime_sock
        resolve_local_ip_mock.return_value = "10.0.0.5"
        get_local_mac_address_mock.return_value = "aa:bb:cc:dd:ee:ff"
        recv_packet_mock.return_value = (("10.0.0.2", 5000), build_discover_message(HOME_COMPUTER_NAME))
        thread_instance = mock.Mock()
        thread_mock.return_value = thread_instance

        stop_values = iter([False, True])
        runtime = MainNodeRuntime(
            config=AppConfig(),
            logger=mock.Mock(),
            should_stop=lambda: next(stop_values),
        )

        result = runtime.run()

        self.assertEqual(
            result,
            DiscoveryResult(
                success=True,
                peer_address="10.0.0.5",
                peer_port=DEFAULT_TCP_PORT,
                source="home_scheduler",
                message="Home scheduler runtime stopped.",
            ),
        )
        set_socket_timeout_mock.assert_called_once()
        create_tcp_listener_mock.assert_called_once()
        self.assertEqual(thread_mock.call_count, 2)
        self.assertEqual(thread_instance.start.call_count, 2)
        recv_packet_mock.assert_called_once()
        safe_close_mock.assert_called_once_with(runtime_sock)
        close_runtime_connections_mock.assert_called_once()
        close_mock.assert_called_once_with(endpoint)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.multicast.send_announce")
    def test_handle_packet_replies_to_discover(
        self,
        send_announce_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        send_announce_mock.return_value = "10.0.0.5"
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME),
            logger=mock.Mock(),
        )

        runtime._handle_packet(
            mock.Mock(),
            ("10.0.0.2", 5000),
            build_discover_message(HOME_COMPUTER_NAME),
        )

        send_announce_mock.assert_called_once()
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.send_message")
    @mock.patch("main_node.runtime.recv_message")
    def test_register_runtime_connection_accepts_register_worker(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        worker_connection = mock.Mock()
        runtime.registry.register_worker.return_value = worker_connection
        client_sock = mock.Mock()
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
        recv_message_mock.return_value = build_register_worker(HOME_COMPUTER_NAME, hardware)

        runtime._register_runtime_connection(client_sock, ("10.0.0.2", 5000), "10.0.0.5")

        runtime.registry.register_worker.assert_called_once_with(
            node_name=HOME_COMPUTER_NAME,
            peer_address="10.0.0.2",
            peer_port=5000,
            hardware=hardware,
            sock=client_sock,
        )
        response = send_message_mock.call_args.args[1]
        self.assertEqual(response.kind, MessageKind.REGISTER_OK)
        self.assertEqual(response.register_ok.scheduler_ip, "10.0.0.5")
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.threading.Thread")
    @mock.patch("main_node.runtime.send_message")
    @mock.patch("main_node.runtime.recv_message")
    def test_register_runtime_connection_accepts_client_join(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        thread_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        runtime.registry.count_workers.return_value = 2
        runtime.registry.count_clients.return_value = 1
        client_connection = mock.Mock()
        client_connection.node_name = HOME_CLIENT_NAME
        runtime.registry.register_client.return_value = client_connection
        thread_instance = mock.Mock()
        thread_mock.return_value = thread_instance
        client_sock = mock.Mock()
        recv_message_mock.return_value = build_client_join(HOME_CLIENT_NAME)

        runtime._register_runtime_connection(client_sock, ("10.0.0.3", 6000), "10.0.0.5")

        runtime.registry.register_client.assert_called_once_with(
            node_name=HOME_CLIENT_NAME,
            peer_address="10.0.0.3",
            peer_port=6000,
            sock=client_sock,
        )
        response = send_message_mock.call_args.args[1]
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.request_id, "join")
        self.assertEqual(response.client_response.worker_count, 2)
        self.assertEqual(response.client_response.client_count, 1)
        thread_mock.assert_called_once()
        thread_instance.start.assert_called_once()
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.network.safe_close")
    @mock.patch("main_node.runtime.send_message")
    @mock.patch("main_node.runtime.recv_message")
    def test_serve_client_connection_replies_to_client_request(
        self,
        recv_message_mock: mock.Mock,
        send_message_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME),
            logger=mock.Mock(),
        )
        runtime.registry = mock.Mock()
        connection = mock.Mock()
        connection.peer_id = "client:home client@10.0.0.3:6000"
        connection.node_name = HOME_CLIENT_NAME
        connection.peer_address = "10.0.0.3"
        connection.peer_port = 6000
        connection.sock = mock.Mock()
        runtime.registry.remove_client.return_value = connection
        recv_message_mock.side_effect = [
            build_client_request(HOME_CLIENT_NAME, "req-1", "text", "hello main node"),
            None,
        ]

        runtime._serve_client_connection(connection)

        runtime.registry.mark_client_request.assert_called_once_with(connection.peer_id)
        response = send_message_mock.call_args.args[1]
        self.assertEqual(response.kind, MessageKind.CLIENT_RESPONSE)
        self.assertEqual(response.client_response.request_id, "req-1")
        self.assertTrue(response.client_response.ok)
        self.assertEqual(response.client_response.message, "")
        self.assertEqual(response.client_response.payload, "")
        self.assertEqual(response.client_response.worker_count, 0)
        self.assertEqual(response.client_response.client_count, 0)
        runtime.registry.remove_client.assert_called_once_with(connection.peer_id)
        safe_close_mock.assert_called_once_with(connection.sock)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.network.safe_close")
    @mock.patch("main_node.runtime.send_message")
    def test_send_heartbeat_once_retries_and_removes_timed_out_worker(
        self,
        send_message_mock: mock.Mock,
        safe_close_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        send_message_mock.side_effect = OSError("broken socket")
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME, heartbeat_retry_count=3),
            logger=mock.Mock(),
        )
        connection = mock.Mock()
        connection.peer_id = "worker:home computer@10.0.0.2:5000"
        connection.node_name = HOME_COMPUTER_NAME
        connection.peer_address = "10.0.0.2"
        connection.peer_port = 5000
        connection.sock = mock.Mock()
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = [connection]
        runtime.registry.remove_worker.return_value = connection

        runtime._send_heartbeat_once()

        self.assertEqual(send_message_mock.call_count, 4)
        runtime.registry.remove_worker.assert_called_once_with(connection.peer_id)
        safe_close_mock.assert_called_once_with(connection.sock)
        self.assertTrue(print_mock.called)

    @mock.patch("builtins.print")
    @mock.patch("main_node.runtime.recv_message")
    @mock.patch("main_node.runtime.send_message")
    @mock.patch("main_node.runtime.build_heartbeat")
    def test_send_heartbeat_once_marks_alive_worker_on_ack(
        self,
        build_heartbeat_mock: mock.Mock,
        send_message_mock: mock.Mock,
        recv_message_mock: mock.Mock,
        print_mock: mock.Mock,
    ) -> None:
        runtime = MainNodeRuntime(
            config=AppConfig(node_name=HOME_SCHEDULER_NAME),
            logger=mock.Mock(),
        )
        connection = mock.Mock()
        connection.peer_id = "worker:home computer@10.0.0.2:5000"
        connection.node_name = HOME_COMPUTER_NAME
        connection.peer_address = "10.0.0.2"
        connection.peer_port = 5000
        connection.sock = mock.Mock()
        runtime.registry = mock.Mock()
        runtime.registry.list_workers.return_value = [connection]
        build_heartbeat_mock.return_value = build_heartbeat(HOME_SCHEDULER_NAME, unix_time_ms=123456)
        recv_message_mock.return_value = build_heartbeat_ok(HOME_COMPUTER_NAME, 123456, received_unix_time_ms=124000)

        runtime._send_heartbeat_once()

        send_message_mock.assert_called_once()
        runtime.registry.mark_heartbeat.assert_called_once_with(connection.peer_id, sent_at=124.0)
        runtime.registry.remove_worker.assert_not_called()
        self.assertTrue(print_mock.called)


if __name__ == "__main__":
    unittest.main()