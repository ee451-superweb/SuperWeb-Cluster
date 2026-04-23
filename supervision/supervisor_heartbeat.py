"""Local IPC heartbeat between the supervisor and the peer it spawned.

The supervisor needs a way to detect that its peer subprocess is still alive
at the Python level, not just at the OS level. A peer that is stuck in a
native CUDA dispatch, deadlocked on a lock, or spinning inside a C extension
keeps its OS process alive indefinitely — ``process.poll()`` returns ``None``,
``wait()`` never returns, and the main cluster-heartbeat path only detects it
after ~4 missed heartbeats AND only for the peer that registered with *this*
host's main node. A second supervisor on another host never owns a registry,
so it gets no signal at all.

Design:

* Supervisor binds a loopback TCP listener on an ephemeral port BEFORE
  spawning the peer, then passes the port to the child via an environment
  variable. Loopback TCP was picked over ``socket.socketpair`` + fd
  inheritance because Windows fd inheritance requires handle-sharing via
  ``socket.share`` / ``fromshare`` that has no POSIX equivalent; a single
  ``accept`` on 127.0.0.1 has identical semantics and needs no per-platform
  branching.
* The peer reads the env var early in bootstrap, connects once, and a daemon
  thread writes a heartbeat byte every ``HEARTBEAT_INTERVAL_SECONDS``. If
  the peer's main thread is stuck holding the GIL, that daemon thread cannot
  run, bytes stop arriving, and the supervisor sees the hang directly.
* The supervisor's watcher blocks on ``recv`` with a timeout slightly larger
  than the heartbeat interval. ``HEARTBEAT_MISS_THRESHOLD`` consecutive
  timeouts while ``process.poll() is None`` means "alive but hung" — the
  supervisor then dumps a py-spy stack and terminates the peer.

Failure modes that DO NOT trigger a kill:
  * Connection refused / env var unset → peer is a legacy build without
    heartbeat support, fall back silently to the exit-code watcher.
  * Peer exits cleanly → listener sees EOF; ``_watch_peer`` handles the
    exit-code logging, heartbeat watcher just exits.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from typing import Optional

HEARTBEAT_PORT_ENV = "SUPERWEB_PEER_HEARTBEAT_PORT"
HEARTBEAT_INTERVAL_SECONDS = 2.0
HEARTBEAT_MISS_THRESHOLD = 4
_PEER_CONNECT_TIMEOUT_SECONDS = 5.0

# Tri-state return from ``wait_for_heartbeat``. A clean EOF (peer closed the
# socket because its process is exiting) must NOT be treated as a miss:
# peer-exits-cleanly and peer-is-hung produce opposite signals here and the
# supervisor keys off them differently. Reported incident 2026-04-21: the
# native CUDA runner returned exit 1, the peer exited, the socket closed,
# and the watcher logged four "misses" in the same millisecond and then
# tried to py-spy an already-dead pid.
HEARTBEAT_OK = "ok"
HEARTBEAT_TIMEOUT = "timeout"
HEARTBEAT_CLOSED = "closed"


class SupervisorHeartbeatListener:
    """Loopback TCP listener owned by the supervisor side of the link.

    One listener, one accepted connection. ``accept`` is called once from the
    watcher thread after the peer is spawned; subsequent reads go through
    ``wait_for_heartbeat``. The listener closes its accept socket as soon as
    the peer connects so a crashed peer cannot leave us with a half-open
    listener that silently swallows future connections.
    """

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._port: int = self._listener.getsockname()[1]
        self._conn: Optional[socket.socket] = None
        self._closed = False

    @property
    def port(self) -> int:
        return self._port

    def accept(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the peer to connect.

        Returns ``True`` when a connection was accepted, ``False`` on timeout
        or after ``close`` has been called.
        """
        if self._closed:
            return False
        self._listener.settimeout(timeout)
        try:
            conn, _ = self._listener.accept()
        except (socket.timeout, TimeoutError):
            return False
        except OSError:
            return False
        finally:
            try:
                self._listener.close()
            except OSError:
                pass
        self._conn = conn
        return True

    def wait_for_heartbeat(self, timeout: float) -> str:
        """Return the read status within ``timeout``.

        ``HEARTBEAT_OK``      — at least one heartbeat byte arrived.
        ``HEARTBEAT_TIMEOUT`` — nothing arrived; could be a hang or a slow
                                interpreter pause. Caller distinguishes via
                                ``process.poll()`` plus a miss counter.
        ``HEARTBEAT_CLOSED``  — the peer closed the socket (clean EOF) or
                                the socket errored. The peer is exiting; the
                                caller must NOT accumulate misses here,
                                because on peer exit the recv returns EOF
                                instantly and would register four misses in
                                a millisecond.
        """
        if self._conn is None:
            return HEARTBEAT_CLOSED
        self._conn.settimeout(timeout)
        try:
            data = self._conn.recv(64)
        except (socket.timeout, TimeoutError):
            return HEARTBEAT_TIMEOUT
        except OSError:
            return HEARTBEAT_CLOSED
        if not data:
            return HEARTBEAT_CLOSED
        return HEARTBEAT_OK

    def close(self) -> None:
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        try:
            self._listener.close()
        except OSError:
            pass


def start_peer_heartbeat(
    port: int,
    *,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    logger: Optional[logging.Logger] = None,
) -> Optional[threading.Event]:
    """Connect to the supervisor and spawn the peer-side heartbeat thread.

    Returns the ``threading.Event`` that stops the thread, or ``None`` if the
    connection could not be established (in which case the peer runs without
    heartbeat — the supervisor's exit-code watcher is the fallback).
    """
    try:
        sock = socket.create_connection(
            ("127.0.0.1", port),
            timeout=_PEER_CONNECT_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        if logger is not None:
            logger.warning(
                "Peer heartbeat disabled: cannot connect to supervisor on 127.0.0.1:%s: %s",
                port,
                exc,
            )
        return None

    sock.settimeout(None)
    stop = threading.Event()

    def _loop() -> None:
        try:
            while not stop.is_set():
                try:
                    sock.sendall(b".")
                except OSError:
                    return
                if stop.wait(interval):
                    return
        finally:
            try:
                sock.close()
            except OSError:
                pass

    thread = threading.Thread(target=_loop, name="peer-heartbeat", daemon=True)
    thread.start()
    return stop


def start_peer_heartbeat_from_env(
    logger: Optional[logging.Logger] = None,
) -> Optional[threading.Event]:
    """Start the peer heartbeat when the supervisor-provided env var is set.

    Called from bootstrap as early as possible when this process is running
    as a supervisor-spawned peer. If the env var is missing (e.g. running
    standalone under pytest or invoked directly by an operator), this is a
    silent no-op — heartbeat is an observability feature, not a requirement.
    """
    port_str = os.environ.get(HEARTBEAT_PORT_ENV)
    if not port_str:
        return None
    try:
        port = int(port_str)
    except ValueError:
        if logger is not None:
            logger.warning(
                "Ignoring invalid %s=%r; peer heartbeat will not start.",
                HEARTBEAT_PORT_ENV,
                port_str,
            )
        return None
    return start_peer_heartbeat(port, logger=logger)
