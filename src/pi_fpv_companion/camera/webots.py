"""WebotsCamera — a Camera source backed by the WebotsArduVehicle image stream.

The ArduPilot Webots controller (libraries/SITL/examples/Webots_Python) streams
the simulated vehicle camera over TCP: a `=HH` header (width, height) followed by
width*height grayscale bytes, repeated per frame. This adapter consumes that
stream and yields `FrameBundle`s, so the *unchanged* production pipeline
(detector → tracker → filter → servo → FC) can run against a real rendered camera
feed driven by ArduPilot SITL flight dynamics.

Frames are emitted as 3-channel BGR (gray replicated) so overlays and the
BGR-expecting detectors behave exactly as on the Pi. Detections are produced by a
separate Detector (e.g. ArucoDetector) — this source carries none inline.
"""
from __future__ import annotations
import socket
import struct
import time
from typing import Iterator

import cv2
import numpy as np

from pi_fpv_companion.camera.base import FrameBundle

_HEADER = struct.Struct("=HH")


class WebotsCamera:
    def __init__(self, host: str = "127.0.0.1", port: int = 5599,
                 connect_timeout_s: float = 30.0) -> None:
        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None

    def open(self) -> None:
        deadline = time.monotonic() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self._host, self._port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                return
            except OSError as e:                 # stream server not up yet
                last_err = e
                time.sleep(0.5)
        raise ConnectionError(
            f"WebotsCamera could not connect to {self._host}:{self._port} "
            f"within {self._connect_timeout_s:.0f}s (is Webots streaming the camera?)"
        ) from last_err

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _recv_exactly(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(min(n - len(buf), 65536))
            if not chunk:
                return None                      # stream closed
            buf += chunk
        return bytes(buf)

    def frames(self) -> Iterator[FrameBundle]:
        if self._sock is None:
            raise RuntimeError("WebotsCamera.open() must be called before frames()")
        while True:
            header = self._recv_exactly(_HEADER.size)
            if header is None:
                return                           # Webots disconnected → end iteration
            width, height = _HEADER.unpack(header)
            payload = self._recv_exactly(width * height)
            if payload is None:
                return
            gray = np.frombuffer(payload, np.uint8).reshape((height, width))
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            yield FrameBundle(image=bgr, width=width, height=height,
                              timestamp=time.monotonic(), detections=[])
