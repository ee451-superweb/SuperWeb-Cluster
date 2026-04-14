"""Minimal mDNS/DNS-SD helpers for superweb-cluster main-node discovery."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

from app.constants import (
    MAIN_NODE_NAME,
    MDNS_QUERY_UNICAST_RESPONSE,
    MDNS_RECORD_TTL,
    MDNS_SERVICE_ROLE,
    MDNS_SERVICE_TYPE,
)

DNS_HEADER = struct.Struct("!HHHHHH")
DNS_QUESTION = struct.Struct("!HH")
DNS_RECORD = struct.Struct("!HHIH")

DNS_FLAG_RESPONSE = 0x8000
DNS_FLAG_AUTHORITATIVE = 0x0400
DNS_FLAG_RESPONSE_AUTHORITATIVE = DNS_FLAG_RESPONSE | DNS_FLAG_AUTHORITATIVE

DNS_CLASS_IN = 0x0001
DNS_CLASS_CACHE_FLUSH = 0x8000
DNS_CLASS_UNICAST_RESPONSE = 0x8000

DNS_TYPE_A = 0x0001
DNS_TYPE_PTR = 0x000C
DNS_TYPE_TXT = 0x0010
DNS_TYPE_SRV = 0x0021


@dataclass(slots=True)
class AnnouncePayload:
    """Parsed main-node announcement contents."""

    host: str
    port: int
    node_name: str


@dataclass(slots=True)
class _DnsHeader:
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
class _DnsQuestion:
    name: str
    qtype: int
    qclass: int

    @property
    def class_code(self) -> int:
        return self.qclass & ~DNS_CLASS_UNICAST_RESPONSE


@dataclass(slots=True)
class _DnsRecord:
    name: str
    rtype: int
    rclass: int
    ttl: int
    value: object


@dataclass(slots=True)
class _DnsMessage:
    header: _DnsHeader
    questions: list[_DnsQuestion]
    answers: list[_DnsRecord]
    authorities: list[_DnsRecord]
    additionals: list[_DnsRecord]


def _normalize_name(name: str) -> str:
    if name == ".":
        return "."
    return f"{name.rstrip('.')}."


def _canonical_name(name: str) -> str:
    return _normalize_name(name).lower()


def _sanitize_label(text: str) -> str:
    cleaned = []
    for char in text.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_", " "}:
            cleaned.append("-")
    label = "".join(cleaned).strip("-")
    if not label:
        label = MAIN_NODE_NAME.replace(" ", "-")
    return label[:63]


def _scheduler_instance_name(node_name: str) -> str:
    return _normalize_name(f"{_sanitize_label(node_name)}.{MDNS_SERVICE_TYPE}")


def _scheduler_host_name(node_name: str) -> str:
    return _normalize_name(f"{_sanitize_label(node_name)}.local.")


def _encode_name(name: str) -> bytes:
    normalized = _normalize_name(name)
    if normalized == ".":
        return b"\x00"

    encoded = bytearray()
    for label in normalized.rstrip(".").split("."):
        label_bytes = label.encode("utf-8")
        if len(label_bytes) > 63:
            raise ValueError(f"DNS label too long: {label}")
        encoded.append(len(label_bytes))
        encoded.extend(label_bytes)
    encoded.append(0)
    return bytes(encoded)


def _decode_name(message: bytes, offset: int) -> tuple[str, int]:
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
                raise ValueError("DNS compression pointer truncated")
            pointer = ((length & 0x3F) << 8) | message[offset + 1]
            if pointer in visited:
                raise ValueError("DNS compression pointer loop detected")
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
            raise ValueError("DNS label truncated")
        labels.append(label.decode("utf-8", errors="replace"))
        offset += length
        if not jumped:
            end_offset = offset

    return (".".join(labels) + "." if labels else "."), end_offset


def _encode_header(
    *,
    flags: int,
    question_count: int,
    answer_count: int,
    authority_count: int,
    additional_count: int,
    query_id: int = 0,
) -> bytes:
    return DNS_HEADER.pack(
        query_id,
        flags,
        question_count,
        answer_count,
        authority_count,
        additional_count,
    )


def _encode_question(name: str, qtype: int, qclass: int) -> bytes:
    return _encode_name(name) + DNS_QUESTION.pack(qtype, qclass)


def _encode_record(name: str, rtype: int, rclass: int, ttl: int, rdata: bytes) -> bytes:
    return _encode_name(name) + DNS_RECORD.pack(rtype, rclass, ttl, len(rdata)) + rdata


def _encode_txt(strings: list[str]) -> bytes:
    data = bytearray()
    for text in strings:
        raw = text.encode("utf-8")
        if len(raw) > 255:
            raise ValueError("TXT value too long")
        data.append(len(raw))
        data.extend(raw)
    return bytes(data)


def _parse_txt(data: bytes) -> list[str]:
    values: list[str] = []
    offset = 0
    while offset < len(data):
        length = data[offset]
        offset += 1
        chunk = data[offset : offset + length]
        if len(chunk) != length:
            raise ValueError("TXT record truncated")
        values.append(chunk.decode("utf-8", errors="replace"))
        offset += length
    return values


def _parse_record(message: bytes, offset: int) -> tuple[_DnsRecord, int]:
    name, offset = _decode_name(message, offset)
    rtype, rclass, ttl, rdlength = DNS_RECORD.unpack_from(message, offset)
    offset += DNS_RECORD.size
    rdata_offset = offset
    rdata = message[rdata_offset : rdata_offset + rdlength]
    if len(rdata) != rdlength:
        raise ValueError("Resource record data truncated")
    offset += rdlength

    if rtype == DNS_TYPE_PTR:
        value, _ = _decode_name(message, rdata_offset)
    elif rtype == DNS_TYPE_SRV:
        if rdlength < 6:
            raise ValueError("SRV record truncated")
        priority, weight, port = struct.unpack_from("!HHH", message, rdata_offset)
        target, _ = _decode_name(message, rdata_offset + 6)
        value = (priority, weight, port, target)
    elif rtype == DNS_TYPE_A:
        if rdlength != 4:
            raise ValueError("A record length invalid")
        value = socket.inet_ntoa(rdata)
    elif rtype == DNS_TYPE_TXT:
        value = _parse_txt(rdata)
    else:
        value = rdata

    return _DnsRecord(name=name, rtype=rtype, rclass=rclass, ttl=ttl, value=value), offset


def _parse_message(message: bytes) -> _DnsMessage | None:
    if len(message) < DNS_HEADER.size:
        return None

    try:
        header = _DnsHeader(*DNS_HEADER.unpack_from(message, 0))
        offset = DNS_HEADER.size

        questions: list[_DnsQuestion] = []
        for _ in range(header.question_count):
            name, offset = _decode_name(message, offset)
            qtype, qclass = DNS_QUESTION.unpack_from(message, offset)
            offset += DNS_QUESTION.size
            questions.append(_DnsQuestion(name=name, qtype=qtype, qclass=qclass))

        def parse_records(count: int) -> tuple[list[_DnsRecord], int]:
            records: list[_DnsRecord] = []
            local_offset = offset_ref[0]
            for _ in range(count):
                record, local_offset = _parse_record(message, local_offset)
                records.append(record)
            offset_ref[0] = local_offset
            return records, local_offset

        offset_ref = [offset]
        answers, _ = parse_records(header.answer_count)
        authorities, _ = parse_records(header.authority_count)
        additionals, _ = parse_records(header.additional_count)
    except (ValueError, struct.error, OSError):
        return None

    return _DnsMessage(
        header=header,
        questions=questions,
        answers=answers,
        authorities=authorities,
        additionals=additionals,
    )


def _txt_lookup(values: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for value in values:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _instance_to_node_name(instance_name: str) -> str:
    normalized = _canonical_name(instance_name)
    suffix = _canonical_name(MDNS_SERVICE_TYPE)
    if normalized.endswith(suffix):
        label = normalized[: -len(suffix)].rstrip(".")
        if label:
            return label
    return MDNS_SERVICE_ROLE


def build_discover_message(node_name: str) -> bytes:
    """Build an mDNS main-node browse query."""

    del node_name

    qclass = DNS_CLASS_IN
    if MDNS_QUERY_UNICAST_RESPONSE:
        qclass |= DNS_CLASS_UNICAST_RESPONSE

    return (
        _encode_header(
            flags=0,
            question_count=1,
            answer_count=0,
            authority_count=0,
            additional_count=0,
        )
        + _encode_question(MDNS_SERVICE_TYPE, DNS_TYPE_PTR, qclass)
    )


def build_announce_message(host: str, port: int, node_name: str) -> bytes:
    """Build an mDNS main-node service announcement."""

    instance_name = _scheduler_instance_name(node_name)
    host_name = _scheduler_host_name(node_name)

    answers = [
        _encode_record(
            MDNS_SERVICE_TYPE,
            DNS_TYPE_PTR,
            DNS_CLASS_IN,
            MDNS_RECORD_TTL,
            _encode_name(instance_name),
        ),
    ]
    additionals = [
        _encode_record(
            instance_name,
            DNS_TYPE_SRV,
            DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
            MDNS_RECORD_TTL,
            struct.pack("!HHH", 0, 0, port) + _encode_name(host_name),
        ),
        _encode_record(
            instance_name,
            DNS_TYPE_TXT,
            DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
            MDNS_RECORD_TTL,
            _encode_txt(
                [
                    f"role={MDNS_SERVICE_ROLE}",
                    f"node={node_name}",
                ]
            ),
        ),
        _encode_record(
            host_name,
            DNS_TYPE_A,
            DNS_CLASS_IN | DNS_CLASS_CACHE_FLUSH,
            MDNS_RECORD_TTL,
            socket.inet_aton(host),
        ),
    ]

    return _encode_header(
        flags=DNS_FLAG_RESPONSE_AUTHORITATIVE,
        question_count=0,
        answer_count=len(answers),
        authority_count=0,
        additional_count=len(additionals),
    ) + b"".join(answers) + b"".join(additionals)


def parse_discover_message(message: bytes) -> bool:
    """Return True when the packet is a main-node service browse query."""

    parsed = _parse_message(message)
    if parsed is None or parsed.header.is_response:
        return False

    target_name = _canonical_name(MDNS_SERVICE_TYPE)
    for question in parsed.questions:
        if (
            _canonical_name(question.name) == target_name
            and question.qtype == DNS_TYPE_PTR
            and question.class_code == DNS_CLASS_IN
        ):
            return True
    return False


def parse_announce_message(message: bytes) -> AnnouncePayload | None:
    """Parse a main-node service announcement into a structured payload."""

    parsed = _parse_message(message)
    if parsed is None or not parsed.header.is_response:
        return None

    records = parsed.answers + parsed.authorities + parsed.additionals
    target_service = _canonical_name(MDNS_SERVICE_TYPE)
    service_instances: list[str] = []
    service_targets: dict[str, tuple[str, int]] = {}
    ipv4_addresses: dict[str, str] = {}
    node_names: dict[str, str] = {}

    for record in records:
        record_name = _canonical_name(record.name)
        if record.rtype == DNS_TYPE_PTR and record_name == target_service and isinstance(record.value, str):
            service_instances.append(_canonical_name(record.value))
            continue

        if record.rtype == DNS_TYPE_SRV and isinstance(record.value, tuple):
            _priority, _weight, port, target = record.value
            service_targets[record_name] = (_canonical_name(target), port)
            continue

        if record.rtype == DNS_TYPE_A and isinstance(record.value, str):
            ipv4_addresses[record_name] = record.value
            continue

        if record.rtype == DNS_TYPE_TXT and isinstance(record.value, list):
            node_name = _txt_lookup(record.value, "node")
            if node_name:
                node_names[record_name] = node_name

    for instance_name in service_instances:
        target = service_targets.get(instance_name)
        if target is None:
            continue
        host_name, port = target
        host = ipv4_addresses.get(host_name)
        if host is None:
            continue
        return AnnouncePayload(
            host=host,
            port=port,
            node_name=node_names.get(instance_name, _instance_to_node_name(instance_name)),
        )

    return None


def describe_discovery_message(message: bytes) -> str:
    """Return a human-readable summary of a superweb-cluster discovery packet."""

    if parse_discover_message(message):
        return f"mDNS PTR query for {MDNS_SERVICE_TYPE}"

    payload = parse_announce_message(message)
    if payload is not None:
        return (
            f"mDNS main-node announcement from {payload.node_name} "
            f"at {payload.host}:{payload.port}"
        )

    parsed = _parse_message(message)
    if parsed is None:
        return "unrecognized binary packet"

    return (
        f"mDNS packet qd={parsed.header.question_count} "
        f"an={parsed.header.answer_count} ar={parsed.header.additional_count}"
    )


def normalize_manual_address(value: str, default_port: int) -> tuple[str, int]:
    """Normalize manual input into host and port."""

    text = value.strip()
    if not text:
        raise ValueError("manual address is empty")

    if ":" not in text:
        return text, default_port

    host, port_text = text.rsplit(":", 1)
    if not host:
        raise ValueError("manual host is empty")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("manual port is invalid") from exc

    return host, port

