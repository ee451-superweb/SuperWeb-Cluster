"""Kickoff configuration defaults."""

from dataclasses import dataclass

from constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_DISCOVERY_ATTEMPTS,
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_DISCOVERY_RETRY_DELAY,
    DEFAULT_DISCOVERY_TIMEOUT,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_HEARTBEAT_RETRY_COUNT,
    DEFAULT_MAX_PROTOBUF_MESSAGE_SIZE,
    DEFAULT_MAIN_NODE_POLL_TIMEOUT,
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_MULTICAST_TTL,
    DEFAULT_RUNTIME_SOCKET_TIMEOUT,
    DEFAULT_TCP_CONNECT_TIMEOUT,
    DEFAULT_TCP_PORT,
)


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for the kickoff workflow."""

    role: str = "discover"
    node_name: str = "node"
    multicast_group: str = DEFAULT_MULTICAST_GROUP
    udp_port: int = DEFAULT_DISCOVERY_PORT
    tcp_port: int = DEFAULT_TCP_PORT
    discovery_timeout: float = DEFAULT_DISCOVERY_TIMEOUT
    discovery_attempts: int = DEFAULT_DISCOVERY_ATTEMPTS
    discovery_retry_delay: float = DEFAULT_DISCOVERY_RETRY_DELAY
    enable_manual_fallback: bool = True
    buffer_size: int = DEFAULT_BUFFER_SIZE
    multicast_ttl: int = DEFAULT_MULTICAST_TTL
    main_node_poll_timeout: float = DEFAULT_MAIN_NODE_POLL_TIMEOUT
    runtime_socket_timeout: float = DEFAULT_RUNTIME_SOCKET_TIMEOUT
    tcp_connect_timeout: float = DEFAULT_TCP_CONNECT_TIMEOUT
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL
    heartbeat_retry_count: int = DEFAULT_HEARTBEAT_RETRY_COUNT
    max_message_size: int = DEFAULT_MAX_PROTOBUF_MESSAGE_SIZE