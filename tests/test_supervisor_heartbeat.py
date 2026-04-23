"""Tests for the supervisor<->peer loopback heartbeat channel.

The heartbeat is the PRIMARY local hang detector for a peer the supervisor
spawned. These tests pin the contracts the supervisor relies on:

  * the listener's ``port`` is a real, accept()able ephemeral port,
  * a peer writer that connects and sends bytes is observable from the
    listener,
  * a peer writer that stops sending (simulating a GIL-held hang) causes the
    listener to time out deterministically,
  * failure modes on the peer side (no env, invalid env, connection refused)
    degrade to "no heartbeat" WITHOUT raising — the peer must keep running
    even when the IPC path breaks, because the supervisor's exit-code
    watcher is still a valid fallback.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import unittest
from unittest import mock

from supervision import supervisor_heartbeat


def _connect_writer(port: int) -> socket.socket:
    """Return a connected client socket (bypasses the heartbeat thread)."""
    return socket.create_connection(("127.0.0.1", port), timeout=2.0)


class SupervisorHeartbeatListenerTests(unittest.TestCase):
    def test_port_is_ephemeral_and_accepts_connections(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        self.assertGreater(listener.port, 0)

        client = _connect_writer(listener.port)
        self.addCleanup(client.close)
        self.assertTrue(listener.accept(timeout=2.0))

    def test_accept_returns_false_on_timeout(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        self.assertFalse(listener.accept(timeout=0.1))

    def test_wait_for_heartbeat_reads_bytes(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        client = _connect_writer(listener.port)
        self.addCleanup(client.close)
        self.assertTrue(listener.accept(timeout=2.0))

        client.sendall(b".")
        self.assertEqual(
            listener.wait_for_heartbeat(timeout=2.0), supervisor_heartbeat.HEARTBEAT_OK
        )

    def test_wait_for_heartbeat_times_out_when_peer_silent(self) -> None:
        # Silent-client simulates a hung peer: socket is open, thread can't
        # run, bytes don't arrive. Must return TIMEOUT (not CLOSED) so the
        # supervisor accumulates misses and eventually declares a hang.
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        client = _connect_writer(listener.port)
        self.addCleanup(client.close)
        self.assertTrue(listener.accept(timeout=2.0))

        start = time.monotonic()
        self.assertEqual(
            listener.wait_for_heartbeat(timeout=0.2),
            supervisor_heartbeat.HEARTBEAT_TIMEOUT,
        )
        # Sanity: we didn't wait way past the budget (i.e. socket timeout
        # actually applied rather than blocking forever).
        self.assertLess(time.monotonic() - start, 2.0)

    def test_wait_for_heartbeat_reports_closed_on_peer_close(self) -> None:
        # Critical distinction: peer exit produces a clean EOF, which the
        # supervisor must NOT treat as a miss. Tested against a real close,
        # not a mock, because the EOF-vs-timeout branch is what the
        # 2026-04-21 CUDA-runner incident exposed.
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        client = _connect_writer(listener.port)
        self.assertTrue(listener.accept(timeout=2.0))

        client.close()
        self.assertEqual(
            listener.wait_for_heartbeat(timeout=1.0),
            supervisor_heartbeat.HEARTBEAT_CLOSED,
        )


class StartPeerHeartbeatTests(unittest.TestCase):
    def test_writer_delivers_bytes_to_listener(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)

        # Short interval so we don't wait multiple seconds per test.
        stop = supervisor_heartbeat.start_peer_heartbeat(listener.port, interval=0.05)
        self.addCleanup(lambda: stop.set() if stop is not None else None)
        self.assertIsNotNone(stop)
        self.assertTrue(listener.accept(timeout=2.0))

        # Two back-to-back wait calls — the writer thread should keep
        # producing even after the first byte has been drained.
        self.assertTrue(listener.wait_for_heartbeat(timeout=1.0))
        self.assertTrue(listener.wait_for_heartbeat(timeout=1.0))

    def test_writer_stop_event_halts_thread(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        stop = supervisor_heartbeat.start_peer_heartbeat(listener.port, interval=0.05)
        self.assertIsNotNone(stop)
        self.assertTrue(listener.accept(timeout=2.0))

        stop.set()
        # Drain anything already in flight, then confirm silence. The writer
        # closes its socket in ``finally``, so the listener should see either
        # a TIMEOUT (no more bytes) or a CLOSED (peer gone) — either one
        # proves the writer stopped. Only HEARTBEAT_OK means it is still
        # sending.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if listener.wait_for_heartbeat(timeout=0.1) != supervisor_heartbeat.HEARTBEAT_OK:
                return
        self.fail("Writer thread kept sending bytes after stop was set")

    def test_writer_returns_none_on_connection_refused(self) -> None:
        # Bind a socket and immediately close it to obtain a definitely-free
        # port, then try to connect there. Must degrade to None, not raise.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        logger = mock.Mock()
        result = supervisor_heartbeat.start_peer_heartbeat(port, logger=logger)
        self.assertIsNone(result)
        logger.warning.assert_called_once()

    def test_thread_is_named_and_daemonic(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        stop = supervisor_heartbeat.start_peer_heartbeat(listener.port, interval=0.05)
        self.addCleanup(lambda: stop.set() if stop is not None else None)
        self.assertIsNotNone(stop)
        self.assertTrue(listener.accept(timeout=2.0))

        matches = [t for t in threading.enumerate() if t.name == "peer-heartbeat"]
        self.assertTrue(matches, "peer-heartbeat thread should be running")
        self.assertTrue(all(t.daemon for t in matches))


class StartPeerHeartbeatFromEnvTests(unittest.TestCase):
    def test_noop_when_env_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(supervisor_heartbeat.HEARTBEAT_PORT_ENV, None)
            self.assertIsNone(supervisor_heartbeat.start_peer_heartbeat_from_env())

    def test_noop_on_non_integer_env(self) -> None:
        logger = mock.Mock()
        with mock.patch.dict(
            os.environ,
            {supervisor_heartbeat.HEARTBEAT_PORT_ENV: "not-a-port"},
            clear=False,
        ):
            self.assertIsNone(supervisor_heartbeat.start_peer_heartbeat_from_env(logger=logger))
        logger.warning.assert_called_once()

    def test_connects_when_env_points_to_listener(self) -> None:
        listener = supervisor_heartbeat.SupervisorHeartbeatListener()
        self.addCleanup(listener.close)
        with mock.patch.dict(
            os.environ,
            {supervisor_heartbeat.HEARTBEAT_PORT_ENV: str(listener.port)},
            clear=False,
        ):
            stop = supervisor_heartbeat.start_peer_heartbeat_from_env()
        self.addCleanup(lambda: stop.set() if stop is not None else None)
        self.assertIsNotNone(stop)
        self.assertTrue(listener.accept(timeout=2.0))


if __name__ == "__main__":
    unittest.main()
