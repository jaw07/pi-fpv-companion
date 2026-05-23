"""Loopback fake Betaflight for integration-testing the MSP backend.

`make_loopback_pair()` returns two pyserial-like endpoints connected back-to-back.
Pass one to BetaflightBackend's `serial_factory`; pass the other to FakeBetaflight.

FakeBetaflight parses inbound MSP frames, responds to MSP_RC / MSP_STATUS, and
captures the AETR stick values from MSP_SET_RAW_RC for later assertion.
"""
from __future__ import annotations
import struct
import threading
from typing import List, Tuple

from pi_fpv_companion.fc.msp import HEADER_REQUEST, HEADER_RESPONSE, MspDecoder, encode

MSP_STATUS = 101
MSP_RC = 105
MSP_SET_RAW_RC = 200


class LoopbackSerial:
    """Minimal pyserial-like endpoint backed by an in-memory buffer.

    Implements the surface BetaflightBackend uses: in_waiting, read(n), write(b), close().
    Paired endpoints share each other's inbound buffers via _peer.
    """

    def __init__(self) -> None:
        self._inbuf = bytearray()
        self._lock = threading.Lock()
        self._peer: "LoopbackSerial | None" = None
        self.closed = False

    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._inbuf)

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        with self._lock:
            data = bytes(self._inbuf[:n])
            del self._inbuf[:n]
            return data

    def write(self, data: bytes) -> int:
        if self._peer is None:
            return 0
        self._peer._deliver(data)
        return len(data)

    def _deliver(self, data: bytes) -> None:
        with self._lock:
            self._inbuf.extend(data)

    def close(self) -> None:
        self.closed = True


def make_loopback_pair() -> Tuple[LoopbackSerial, LoopbackSerial]:
    a, b = LoopbackSerial(), LoopbackSerial()
    a._peer = b
    b._peer = a
    return a, b


class FakeBetaflight:
    """Receives requests, replies as a Betaflight FC would, captures SET_RAW_RC writes."""

    def __init__(self, serial_endpoint: LoopbackSerial) -> None:
        self._serial = serial_endpoint
        self._decoder = MspDecoder(accept=HEADER_REQUEST)
        self.rc_channels: List[int] = [1500] * 18
        self.armed: bool = False
        self.received_raw_rc: List[Tuple[int, int, int, int]] = []

    def pump(self) -> None:
        """Drain pending bytes from the backend, dispatch frames, send responses."""
        n = self._serial.in_waiting
        if n <= 0:
            return
        data = self._serial.read(n)
        for cmd, payload in self._decoder.feed(data):
            self._handle(cmd, payload)

    def _handle(self, cmd: int, payload: bytes) -> None:
        if cmd == MSP_RC:
            resp = struct.pack(f"<{len(self.rc_channels)}H", *self.rc_channels)
            self._serial.write(encode(MSP_RC, resp, direction=HEADER_RESPONSE))
        elif cmd == MSP_STATUS:
            cycle_time = 500
            i2c_err = 0
            sensors = 0
            flag = 0x1 if self.armed else 0x0
            current_set = 0
            resp = struct.pack("<HHHIB", cycle_time, i2c_err, sensors, flag, current_set)
            self._serial.write(encode(MSP_STATUS, resp, direction=HEADER_RESPONSE))
        elif cmd == MSP_SET_RAW_RC:
            # First 4 channels in AETR order; ignore any additional ones for our tests
            if len(payload) >= 8:
                a, e, t, r = struct.unpack_from("<HHHH", payload, 0)
                self.received_raw_rc.append((a, e, t, r))
