"""Helpers for moving float32 vectors across protobuf byte fields."""

from __future__ import annotations

import array
import sys


def pack_float32_values(values: list[float]) -> bytes:
    """Encode Python floats as little-endian float32 bytes."""

    packed = array.array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def unpack_float32_bytes(raw: bytes) -> list[float]:
    """Decode little-endian float32 bytes into Python floats."""

    if len(raw) % 4 != 0:
        raise ValueError("float32 payload byte length must be a multiple of 4")

    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()
