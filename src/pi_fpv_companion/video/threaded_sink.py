"""Run a video sink's render on its own thread, off the control loop.

On the Pi Zero 2W the main loop was pinned at ~7 Hz: capture -> track -> guidance ->
render all ran serially on one core while the other cores sat idle. The render
(overlay draw + BGR->RGB565 convert + framebuffer write) is ~20-35 ms of that, and
it's all in C extensions (cv2 / numpy / the framebuffer memcpy) that RELEASE the GIL
— so moving it to a second thread genuinely parallelises onto an idle core and frees
the control loop to run at the camera/guidance rate.

`ThreadedSink` wraps any sink with the Pipeline status-callback `.show()` interface.
`.show()` stashes the latest frame and returns immediately; a render thread renders
the most recent frame and DROPS stale ones (latest-wins) — for an FPV feed you want
the freshest frame at lowest latency, never a growing backlog.

Safety / correctness:
  - The flight recorder is NOT here — it runs inline on the control loop in main's
    on_status, so telemetry is recorded every tick regardless of dropped video frames.
  - `frame.image` from picamera2's make_array() is a private copy of the camera buffer
    (picamera2 copies so the buffer can be recycled), and the control loop only READS
    it after handoff, so passing the reference across threads is safe; the wrapped sink
    copies again before it draws. No extra copy needed here.
  - A render exception is logged and swallowed — a bad frame must never kill the render
    thread or the process.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ThreadedSink:
    """Wrap a sink so its `.show()` render runs on a background thread, dropping
    stale frames (latest-wins). Forwards `open()`/`close()` to the wrapped sink."""

    def __init__(self, sink) -> None:
        self._sink = sink
        self._lock = threading.Lock()
        self._latest: Optional[tuple] = None   # newest unrendered payload, or None
        self._wake = threading.Event()
        self._stop = False
        self._dropped = 0
        self._rendered = 0
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        self._thread.start()

    def open(self) -> None:
        # The wrapped framebuffer sink opens lazily on first show() (which now runs on
        # the render thread), but forward an explicit open() if the caller makes one.
        if hasattr(self._sink, "open"):
            self._sink.open()

    def show(self, target, intent, gated, switch, armed, frame, tracks=None) -> None:
        """Non-blocking: replace the pending frame with this one and wake the renderer.
        If a frame was already pending (renderer still busy), it is dropped."""
        payload = (target, intent, gated, switch, armed, frame, tracks)
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            self._latest = payload
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                break
            with self._lock:
                payload = self._latest
                self._latest = None
            if payload is None:
                continue
            try:
                self._sink.show(*payload)
                self._rendered += 1
            except Exception:
                # Never let one bad frame kill the render thread or crash the process.
                logger.exception("render sink failed; dropping frame")

    @property
    def stats(self) -> tuple:
        """(rendered, dropped) frame counts — for diagnostics."""
        return (self._rendered, self._dropped)

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)
        if hasattr(self._sink, "close"):
            self._sink.close()
