"""Counter-based deterministic float32 chunk generation helpers."""

from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

MASK64 = 0xFFFFFFFFFFFFFFFF
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_MIX1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MIX2 = 0x94D049BB133111EB


def _splitmix64_value(counter: int, seed: int) -> int:
    """Return one deterministic 64-bit SplitMix64 value."""

    z = (seed + ((counter + 1) * _SPLITMIX64_GAMMA)) & MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX64_MIX1) & MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX64_MIX2) & MASK64
    z ^= z >> 31
    return z & MASK64


def _float32_word_from_value(value: int) -> int:
    """Map one mixed integer value to one finite float32 bit-pattern."""

    lower = value & 0xFFFFFFFF
    sign = (lower & 0x1) << 31
    mantissa = (lower >> 1) & 0x007FFFFF
    return sign | 0x3F000000 | mantissa


def float32_chunk_from_counter_python(count: int, start_index: int, seed: int) -> bytearray:
    """Generate one deterministic chunk using only the Python standard library."""

    payload = bytearray(count * 4)
    words = memoryview(payload).cast("I")
    try:
        for index in range(count):
            words[index] = _float32_word_from_value(_splitmix64_value(start_index + index, seed))
    finally:
        words.release()
    return payload


def float32_chunk_from_counter_numpy(count: int, start_index: int, seed: int) -> bytes:
    """Generate one deterministic chunk with NumPy vectorized integer math."""

    assert np is not None
    counters = np.arange(start_index, start_index + count, dtype=np.uint64)
    values = np.uint64(seed) + (counters + np.uint64(1)) * np.uint64(_SPLITMIX64_GAMMA)
    values = np.bitwise_xor(values, np.right_shift(values, np.uint64(30)))
    values = values * np.uint64(_SPLITMIX64_MIX1)
    values = np.bitwise_xor(values, np.right_shift(values, np.uint64(27)))
    values = values * np.uint64(_SPLITMIX64_MIX2)
    values = np.bitwise_xor(values, np.right_shift(values, np.uint64(31)))

    lower = values.astype(np.uint32, copy=False)
    words = np.bitwise_or(
        np.left_shift(np.bitwise_and(lower, np.uint32(0x1)), np.uint32(31)),
        np.bitwise_or(
            np.uint32(0x3F000000),
            np.bitwise_and(np.right_shift(lower, np.uint32(1)), np.uint32(0x007FFFFF)),
        ),
    )
    return words.astype("<u4", copy=False).tobytes(order="C")


def float32_chunk_from_counter(count: int, start_index: int, seed: int) -> bytes | bytearray:
    """Generate one float32 chunk using the fastest implementation available."""

    if np is not None:
        return float32_chunk_from_counter_numpy(count, start_index, seed)
    return float32_chunk_from_counter_python(count, start_index, seed)
