"""Minimal mDNS query/reply helpers for standalone network testing."""

from __future__ import annotations

import socket
import struct
import sys
from dataclasses import dataclass

MDNS_GROUP = "224.0.0.251"
MDNS_PORT = 5353
DEFAULT_SERVICE_NAME = "_superweb-cluster._tcp.local."

DNS_HEADER = struct.Struct("!HHHHHH")
DNS_QUESTION = struct.Struct("!HH")
DNS_RECORD = struct.Struct("!HHIH")

DNS_FLAG_RESPONSE = 0x8000
DNS_FLAG_AUTHORITATIVE = 0x0400
DNS_FLAG_RESPONSE_AUTHORITATIVE = DNS_FLAG_RESPONSE | DNS_FLAG_AUTHORITATIVE

DNS_CLASS_IN = 0x0001
DNS_CLASS_UNICAST_RESPONSE = 0x8000
DNS_CLASS_CACHE_FLUSH = 0x8000

DNS_TYPE_A = 0x0001
DNS_TYPE_PTR = 0x000C
DNS_TYPE_TXT = 0x0010
DNS_TYPE_SRV = 0x0021


@dataclass(slots=True)
class DnsHeader:
    query_id: int
    flags: int
    question_count: int
    answer_count: int
    authority_count: int
    additional_count: int

    @property
    def is_response(self) -> bool:
        return bool(self.flags & DNS_FLAG_RESPONSE)


@dataclass(slots=True)
class DnsQuestion:
    name: str
    qtype: int
    qclass: int

    @property
    def class_code(self) -> int:
        return self.qclass & ~DNS_CLASS_UNICAST_RESPONSE


@dataclass(slots=True)
class DnsRecord:
    name: str
    rtype: int
    rclass: int
    ttl: int
    value: object


@dataclass(slots=True)
class ParsedMessage:
    header: DnsHeader
    questions: list[DnsQuestion]
    answers: list[DnsRecord]
    authorities: list[DnsRecord]
    additionals: list[DnsRecord]


@dataclass(slots=True)
class Announcement:
    instance_name: str
    host_name: str
    host_ip: str
    port: int
    txt_values: list[str]


def normalize_name(name: str) -> str:
    """Return a DNS name with a trailing dot."""

    if name == ".":
        return "."
    return f"{name.rstrip('.')}."


def canonical_name(name: str) -> str:
    """Return a lowercase DNS name with trailing dot."""

    return normalize_name(name).lower()


def sanitize_label(text: str) -> str:
    """Map a user-facing name to a DNS-safe single label."""

    cleaned: list[str] = []
    for char in text.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_", " "}:
            cleaned.append("-")
    label = "".join(cleaned).strip("-")
    if not label:
        label = "node"
    return label[:63]


def make_instance_name(label: str, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """Return a DNS-SD instance name under the target service."""

    return normalize_name(f"{sanitize_label(label)}.{normalize_name(service_name)}")


def make_host_name(label: str) -> str:
    """Return a `.local.` host name."""

    return normalize_name(f"{sanitize_label(label)}.local.")


def encode_name(name: str) -> bytes:
    """Encode a DNS name without compression."""

    normalized = normalize_name(name)
    if normalized == ".":
        return b"\x00"

    data = bytearray()
    for label in normalized.rstrip(".").split("."):
        raw = label.encode("utf-8")
        if len(raw) > 63:
            raise ValueError(f"label too long: {label}")
        data.append(len(raw))
        data.extend(raw)
    data.append(0)
    return bytes(data)


def decode_name(message: bytes, offset: int) -> tuple[str, int]:
    """Decode a possibly compressed DNS name."""

    labels: list[str] = []
    end_offset = offset
    jumped = False
    visited: set[int] = set()

    while True:
        if offset >= len(message):
            raise ValueError("DNS name exceeds packet length")

        length = message[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(message):
                raise ValueError("truncated DNS compression pointer")
            pointer = ((length & 0x3F) << 8) | message[offset + 1]
            if pointer in visited:
                raise ValueError("DNS compression loop detected")
            visited.add(pointer)
            if not jumped:
                end_offset = offset + 2
                jumped = True
            offset = pointer
            continue

        if length == 0:
            if not jumped:
                end_offset = offset + 1
            break

        offset += 1
        label = message[offset : offset + length]
        if len(label) != length:
            raise ValueError("truncated DNS label")
        labels.append(label.decode("utf-8", errors="replace"))
        offset += length
        if not jumped:
            end_offset = offset

    return (".".join(labels) + "." if labels else "."), end_offset


def encode_txt(values: list[str]) -> bytes:
    """Encode TXT strings into wire format."""

    data = bytearray()
    for value in values:
        raw = value.encode("utf-8")
        if len(raw) > 255:
            raise ValueError("TXT entry too long")
        data.append(len(raw))
        data.extend(raw)
    return bytes(data)


def parse_txt(data: bytes) -> list[str]:
    """Parse TXT wire data into strings."""

    values: list[str] = []
    offset = 0
    while offset < len(data):
        size = data[offset]
        offset += 1
        chunk = data[offset : offset + size]
        if len(chunk) != size:
            raise ValueError("truncated TXT record")
        values.append(chunk.decode("utf-8", errors="replace"))
        offset += size
    return values


def encode_header(
    *,
    flags: int,
    question_count: int,
    answer_count: int,
    authority_count: int,
    additional_count: int,
    query_id: int = 0,
) -> bytes:
    """Encode a DNS header."""

    return DNS_HEADER.pack(
        query_id,
        flags,
        question_count,
        answer_count,
        authority_count,
        additional_count,
    )


def encode_question(name: str, qtype: int, qclass: int) -> bytes:
    """Encode a DNS question."""

    return encode_name(name) + DNS_QUESTION.pack(qtype, qclass)


def encode_record(name: str, rtype: int, rclass: int, ttl: int, rdata: bytes) -> bytes:
    """Encode a DNS resource record."""

    return encode_name(name) + DNS_RECORD.pack(rtype, rclass, ttl, len(rdata)) + rdata


def parse_record(message: bytes, offset: int) -> tuple[DnsRecord, int]:
    """Parse one DNS resource record from a packet."""

    name, offset = decode_name(message, offset)
    rtype, rclass, ttl, rdlength = DNS_RECORD.unpack_from(message, offset)
    offset += DNS_RECORD.size
    rdata_offset = offset
    rdata = message[rdata_offset : rdata_offset + rdlength]
    if len(rdata) != rdlength:
        raise ValueError("truncated resource record")
    offset += rdlength

    if rtype == DNS_TYPE_PTR:
        value, _ = decode_name(message, rdata_offset)
    elif rtype == DNS_TYPE_SRV:
        if rdlength < 6:
            raise ValueError("truncated SRV record")
        priority, weight, port = struct.unpack_from("!HHH", message, rdata_offset)
        target, _ = decode_name(message, rdata_offset + 6)
        value = (priority, weight, port, target)
    elif rtype == DNS_TYPE_A:
        if rdlength != 4:
            raise ValueError("invalid A record length")
        value = socket.inet_ntoa(rdata)
    elif rtype == DNS_TYPE_TXT:
        value = parse_txt(rdata)
    else:
        value = rdata

    return DnsRecord(name=name, rtype=rtype, rclass=rclass, ttl=ttl, value=value), offset


def parse_message(message: bytes) -> ParsedMessage | None:
    """Parse a minimal DNS packet."""

    if len(message) < DNS_HEADER.size:
        return None

    try:
        header = DnsHeader(*DNS_HEADER.unpack_from(message, 0))
        offset = DNS_HEADER.size

        questions: list[DnsQuestion] = []
        for _ in range(header.question_count):
            name, offset = decode_name(message, offset)
            qtype, qclass = DNS_QUESTION.unpack_from(message, offset)
            offset += DNS_QUESTION.size
            questions.append(DnsQuestion(name=name, qtype=qtype, qclass=qclass))

        def parse_records(count: int, start: int) -> tuple[list[DnsRecord], int]:
            records: list[DnsRecord] = []
            current = start
            for _ in range(count):
                record, current = parse_record(message, current)
                records.append(record)
            return records, current

        answers, offset = parse_records(header.answer_count, offset)
        authorities, offset = parse_records(header.authority_count, offset)
        additionals, offset = parse_records(header.additional_count, offset)
    except (ValueError, struct.error, OSError):
        return None

    return ParsedMessage(
        header=header,
        questions=questions,
        answers=answers,
        authorities=authorities,
        additionals=additionals,
    )


def build_ptr_query(service_name: str = DEFAULT_SERVICE_NAME, *, request_unicast_response: bool = True) -> bytes:
    """Build a PTR browse query for the target service."""

    qclass = DNS_CLASS_IN
    if request_unicast_response:
        qclass |= DNS_CLASS_UNICAST_RESPONSE

    return encode_header(
        flags=0,
        question_count=1,
        answer_count=0,
        authority_count=0,
        additional_count=0,
    ) + encode_question(service_name, DNS_TYPE_PTR, qclass)


def build_announcement(
    *,
    service_name: str,
    instance_name: str,
    host_name: str,
    host_ip: str,
    service_port: int,
    txt_values: list[str] | None = None,
    ttl: int = 120,
) -> bytes:
    """Build a compact service announcement with PTR, SRV, TXT, and A records."""

    if txt_values is None:
        txt_values = []

    answer = encode_record(
        service_name,
        DNS_TYPE_PTR,
        DNS_CLASS_IN,
        ttl,
        encode_name(instance_name),
    )
    srv = encode_record(
        instance_name,
        DNS_TYPE_SRV,
        DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
        ttl,
        struct.pack("!HHH", 0, 0, service_port) + encode_name(host_name),
    )
    txt = encode_record(
        instance_name,
        DNS_TYPE_TXT,
        DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
        ttl,
        encode_txt(txt_values),
    )
    address = encode_record(
        host_name,
        DNS_TYPE_A,
        DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
        ttl,
        socket.inet_aton(host_ip),
    )

    return (
        encode_header(
            flags=DNS_FLAG_RESPONSE_AUTHORITATIVE,
            question_count=0,
            answer_count=1,
            authority_count=0,
            additional_count=3,
        )
        + answer
        + srv
        + txt
        + address
    )


def packet_requests_service(message: bytes, service_name: str = DEFAULT_SERVICE_NAME) -> bool:
    """Return whether a packet asks for the target PTR service."""

    parsed = parse_message(message)
    if parsed is None or parsed.header.is_response:
        return False

    target = canonical_name(service_name)
    for question in parsed.questions:
        if (
            canonical_name(question.name) == target
            and question.qtype == DNS_TYPE_PTR
            and question.class_code == DNS_CLASS_IN
        ):
            return True
    return False


def parse_announcement(message: bytes, service_name: str = DEFAULT_SERVICE_NAME) -> Announcement | None:
    """Extract the first matching announcement from a response packet."""

    parsed = parse_message(message)
    if parsed is None or not parsed.header.is_response:
        return None

    target_service = canonical_name(service_name)
    records = parsed.answers + parsed.authorities + parsed.additionals

    service_instances: list[str] = []
    service_targets: dict[str, tuple[str, int]] = {}
    ipv4_addresses: dict[str, str] = {}
    txt_by_instance: dict[str, list[str]] = {}

    for record in records:
        record_name = canonical_name(record.name)

        if record.rtype == DNS_TYPE_PTR and record_name == target_service and isinstance(record.value, str):
            service_instances.append(canonical_name(record.value))
            continue

        if record.rtype == DNS_TYPE_SRV and isinstance(record.value, tuple):
            _priority, _weight, port, target = record.value
            service_targets[record_name] = (canonical_name(target), port)
            continue

        if record.rtype == DNS_TYPE_A and isinstance(record.value, str):
            ipv4_addresses[record_name] = record.value
            continue

        if record.rtype == DNS_TYPE_TXT and isinstance(record.value, list):
            txt_by_instance[record_name] = record.value

    for instance_name in service_instances:
        host_target = service_targets.get(instance_name)
        if host_target is None:
            continue
        host_name, port = host_target
        host_ip = ipv4_addresses.get(host_name)
        if host_ip is None:
            continue
        return Announcement(
            instance_name=instance_name,
            host_name=host_name,
            host_ip=host_ip,
            port=port,
            txt_values=txt_by_instance.get(instance_name, []),
        )

    return None


def resolve_local_ip(remote_host: str = MDNS_GROUP, remote_port: int = MDNS_PORT) -> str:
    """Best-effort local IPv4 selection."""

    probe: socket.socket | None = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect((remote_host, remote_port))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        safe_close(probe)


def create_sender_socket(
    *,
    timeout: float,
    interface_ip: str = "",
    ttl: int = 255,
) -> socket.socket:
    """Create an mDNS sender socket bound to an ephemeral port."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.bind(("", 0))
    sock.settimeout(timeout)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("B", ttl))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if interface_ip:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(interface_ip))
    return sock


def create_listener_socket(
    *,
    timeout: float,
    interface_ip: str = "",
) -> tuple[socket.socket, bytes]:
    """Create an mDNS listener bound to UDP/5353 and joined to the multicast group."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if sys.platform == "darwin" and hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("", MDNS_PORT))
    membership_interface = interface_ip or "0.0.0.0"
    membership = struct.pack(
        "4s4s",
        socket.inet_aton(MDNS_GROUP),
        socket.inet_aton(membership_interface),
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.settimeout(timeout)
    return sock, membership


def drop_membership(sock: socket.socket, membership: bytes | None) -> None:
    """Leave the joined multicast group if possible."""

    if membership is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, membership)
    except OSError:
        return


def safe_close(sock: socket.socket | None) -> None:
    """Close a socket without surfacing cleanup errors."""

    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        return
