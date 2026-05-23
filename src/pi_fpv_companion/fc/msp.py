"""MSP v1 framing.

Frame layout:
    $ M dir size cmd payload[size] checksum

  dir       = '<' (request, host -> FC) or '>' (response, FC -> host)
  size      = u8, payload length in bytes
  cmd       = u8, command id
  payload   = size bytes
  checksum  = XOR of size, cmd, and each payload byte

This module only does framing. Command IDs and payload semantics live with the
backend that uses them.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple


HEADER_REQUEST = b"$M<"
HEADER_RESPONSE = b"$M>"


def _checksum(size: int, cmd: int, payload: bytes) -> int:
    csum = size ^ cmd
    for b in payload:
        csum ^= b
    return csum & 0xFF


def encode(cmd: int, payload: bytes = b"", direction: bytes = HEADER_REQUEST) -> bytes:
    """Encode one MSP v1 frame. Direction defaults to request ($M<)."""
    size = len(payload)
    return direction + bytes([size, cmd]) + payload + bytes([_checksum(size, cmd, payload)])


class MspDecoder:
    """Stateful byte-stream decoder. Feed bytes, get back complete (cmd, payload) frames.

    Tolerates partial frames at buffer boundaries and recovers from corruption by
    re-syncing on the next '$' byte after a checksum mismatch.
    """

    def __init__(self, accept: bytes = HEADER_RESPONSE) -> None:
        if len(accept) != 3 or not accept.startswith(b"$M"):
            raise ValueError(f"unexpected direction header: {accept!r}")
        self._accept = accept
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[Tuple[int, bytes]]:
        self._buf.extend(data)
        out: List[Tuple[int, bytes]] = []
        while True:
            idx = self._buf.find(self._accept)
            if idx < 0:
                # Keep at most the last 2 bytes — might be the start of a header.
                if len(self._buf) > 2:
                    self._buf = self._buf[-2:]
                break
            if idx > 0:
                del self._buf[:idx]                 # drop garbage before header
            if len(self._buf) < 6:                  # need header(3) + size + cmd + checksum
                break
            size = self._buf[3]
            total = 5 + size + 1
            if len(self._buf) < total:
                break
            cmd = self._buf[4]
            payload = bytes(self._buf[5 : 5 + size])
            checksum = self._buf[5 + size]
            if _checksum(size, cmd, payload) == checksum:
                out.append((cmd, payload))
                del self._buf[:total]
            else:
                # Bad frame: drop the '$' and re-sync.
                del self._buf[0]
        return out
