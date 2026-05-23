"""MJPEG-over-HTTP preview sink.

Serves the composited pipeline frames (camera image + bbox + HUD overlay) as a
multipart MJPEG stream so you can watch the live feed in any browser over the
network — `http://<pi-ip>:<port>/`.

Dev / bring-up aid, not for flight: JPEG-encoding every frame costs CPU
(~5-15 ms at 720×576 q70 on a Zero 2W) and the analog CVBS path is the real
output. Conforms to the same status-callback signature as LiveViewer /
FramebufferSink, so it's a drop-in `on_status` for the Pipeline.
"""
from __future__ import annotations
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import GuidanceIntent, SwitchState, Target
from pi_fpv_companion.video.overlay import draw_overlay


class _LatestFrame:
    """Single-slot latest-JPEG holder with a sequence number for change-wait."""

    def __init__(self) -> None:
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._cv = threading.Condition()

    def publish(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg = jpeg
            self._seq += 1
            self._cv.notify_all()

    def wait_newer(self, last_seq: int, timeout: float = 5.0):
        with self._cv:
            if self._seq == last_seq:
                self._cv.wait(timeout=timeout)
            return self._jpeg, self._seq


_INDEX_HTML = (
    b"<!doctype html><html><head><title>pi-fpv-companion</title>"
    b"<style>body{margin:0;background:#111;display:flex;justify-content:center;"
    b"align-items:center;height:100vh}"
    # Upscale the (downscaled-for-bandwidth) frames to fill the viewport,
    # preserving aspect ratio. Keeps the wire small but the display large.
    b"img{width:100vw;height:100vh;object-fit:contain}</style>"
    b"</head><body><img src='/stream.mjpg'></body></html>"
)


def _make_handler(latest: _LatestFrame):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass  # silence per-request logging

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)
                return
            if self.path not in ("/stream.mjpg", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            seq = -1
            try:
                while True:
                    jpeg, seq = latest.wait_newer(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


class MjpegStreamSink:
    def __init__(
        self, host: str = "0.0.0.0", port: int = 8080, jpeg_quality: int = 70,
        max_fps: float = 0.0, scale: float = 1.0,
    ) -> None:
        self._host = host
        self._port = port
        self._quality = int(jpeg_quality)
        # Bandwidth controls for slow links (WiFi): cap publish rate and/or
        # downscale before JPEG-encoding. 0/1.0 = unchanged.
        self._min_interval = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        self._scale = float(scale)
        self._last_pub = 0.0
        self._latest = _LatestFrame()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        if self._server is not None:
            return
        self._server = ThreadingHTTPServer(
            (self._host, self._port), _make_handler(self._latest)
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="mjpeg-http"
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def show(
        self,
        target: Optional[Target],
        intent: GuidanceIntent,
        gated: GateResult,
        switch: SwitchState,
        armed: bool,
        frame: FrameBundle,
    ) -> None:
        if self._server is None:
            self.open()
        # Frame-rate cap: drop frames we don't have bandwidth to send.
        if self._min_interval:
            now = time.monotonic()
            if (now - self._last_pub) < self._min_interval:
                return
            self._last_pub = now
        img = frame.image.copy()
        draw_overlay(img, target, intent, switch, armed, gated)
        if self._scale != 1.0:
            img = cv2.resize(img, None, fx=self._scale, fy=self._scale,
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if ok:
            self._latest.publish(buf.tobytes())
