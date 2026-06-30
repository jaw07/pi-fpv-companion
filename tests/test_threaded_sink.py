"""Tests for ThreadedSink: the render runs on a background thread, drops stale
frames (latest-wins), forwards open/close, and survives a render exception."""
import threading
import time

from pi_fpv_companion.video.threaded_sink import ThreadedSink


def _wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


class FakeSink:
    """Records each show() by its `target` arg. Optionally blocks the FIRST show()
    on an event so we can deterministically stack up frames behind a slow render."""

    def __init__(self, block_first=False, raise_on=None):
        self.calls = []
        self.opened = False
        self.closed = False
        self.block_first = block_first
        self.raise_on = raise_on
        self.entered_first = threading.Event()
        self.release_first = threading.Event()
        self._first = True
        self._lock = threading.Lock()

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def show(self, target, intent, gated, switch, armed, frame, tracks=None):
        if self.block_first and self._first:
            self._first = False
            self.entered_first.set()
            self.release_first.wait(2.0)
        if self.raise_on is not None and target == self.raise_on:
            raise RuntimeError("boom")
        with self._lock:
            self.calls.append(target)


def _payload(name):
    # (target, intent, gated, switch, armed, frame, tracks)
    return (name, None, None, None, False, None, None)


def test_show_forwards_to_sink():
    fs = FakeSink()
    ts = ThreadedSink(fs)
    ts.show(*_payload("A"))
    assert _wait_until(lambda: fs.calls == ["A"])
    ts.close()
    assert fs.closed is True


def test_latest_wins_drops_stale_frames():
    fs = FakeSink(block_first=True)
    ts = ThreadedSink(fs)

    ts.show(*_payload("A"))                       # render thread picks A, blocks in show()
    assert fs.entered_first.wait(2.0)
    ts.show(*_payload("B"))                       # queued behind A...
    ts.show(*_payload("C"))                       # ...B replaced by C (B dropped)
    fs.release_first.set()                        # let A finish; renderer then takes C

    assert _wait_until(lambda: fs.calls == ["A", "C"])
    rendered, dropped = ts.stats
    assert "B" not in fs.calls                    # the stale frame was dropped
    assert dropped >= 1
    ts.close()


def test_render_exception_does_not_kill_thread():
    fs = FakeSink(raise_on="boom")
    ts = ThreadedSink(fs)
    ts.show(*_payload("boom"))                    # this render raises (logged, swallowed)
    ts.show(*_payload("ok"))                      # thread must still be alive to render this
    assert _wait_until(lambda: "ok" in fs.calls)
    assert ts._thread.is_alive()
    ts.close()


def test_open_close_forwarded():
    fs = FakeSink()
    ts = ThreadedSink(fs)
    ts.open()
    assert fs.opened is True
    ts.close()
    assert fs.closed is True
    assert not ts._thread.is_alive()
