"""MJPEG-over-HTTP sink — BENCH USE ONLY.

Serves the composited frame (bbox + HUD overlay, identical to the TV-out image)
as a browser-viewable MJPEG stream:

    http://<pi>:<port>/          tiny HTML page wrapping the stream
    http://<pi>:<port>/stream    multipart/x-mixed-replace MJPEG
    http://<pi>:<port>/snapshot.jpg   single latest frame

This exists for FC-less / VTX-less bench rigs (config: video.web_stream_port).
It is NOT a flight path: the flight video output is the analog composite via
FramebufferSink, and streaming costs CPU the flight pipeline needs. The server
keeps only the LATEST frame (slow clients drop frames, they never backpressure
the pipeline) and show() is rate-limited to web_stream_fps.
"""
from __future__ import annotations
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import FilteredTarget, GuidanceIntent, SwitchState
from pi_fpv_companion.video.overlay import draw_overlay

_log = logging.getLogger(__name__)

_INDEX_HTML = b"""<!doctype html>
<html><head><title>pi-fpv-companion bench stream</title>
<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style></head>
<body><img src="/stream" alt="live"></body></html>
"""


class MjpegStreamSink:
    """Pipeline status-callback sink that publishes the overlay-composited frame
    over HTTP as MJPEG. Same `show()` signature as FramebufferSink."""

    def __init__(self, port: int = 8080, quality: int = 80, max_fps: float = 15.0) -> None:
        self._quality = int(quality)
        self._min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_pub = 0.0
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        sink = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):   # quiet: journald doesn't need per-GET lines
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML)
                elif self.path == "/snapshot.jpg":
                    jpeg = sink._latest()
                    if jpeg is None:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        last_seq = -1
                        while True:
                            jpeg, last_seq = sink._wait_frame(last_seq)
                            if jpeg is None:
                                continue
                            self.wfile.write(b"--frame\r\n"
                                             b"Content-Type: image/jpeg\r\n"
                                             b"Content-Length: " +
                                             str(len(jpeg)).encode() + b"\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass          # client went away — normal
                else:
                    self.send_error(404)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server(("0.0.0.0", port), Handler)
        self.port = self._server.server_address[1]   # real port (0 = OS-assigned, for tests)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True, name="mjpeg-stream")
        self._thread.start()
        _log.info("MJPEG bench stream on http://0.0.0.0:%d/ (quality=%d)",
                  self.port, self._quality)

    # ---- frame exchange (pipeline thread -> client threads) ----

    def _latest(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg

    def _wait_frame(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._jpeg, self._seq

    # ---- Pipeline sink interface ----

    def show(
        self,
        target: Optional[FilteredTarget],
        intent: GuidanceIntent,
        gated: GateResult,
        switch: SwitchState,
        armed: bool,
        frame: FrameBundle,
        tracks=None,
    ) -> None:
        now = time.monotonic()
        if now - self._last_pub < self._min_period:
            return
        self._last_pub = now
        import cv2  # lazy, matching the rest of the video stack
        img = frame.image.copy()
        draw_overlay(img, target, intent, switch, armed, gated, tracks)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
