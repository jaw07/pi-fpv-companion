"""Measure sensor-to-userspace frame AGE, and whether it grows into a standing queue.

picamera2's capture_request() hands back completed requests FIFO. With
buffer_count=4 the camera can have up to 4 completed frames queued; if the
consumer ever falls behind by k frames it keeps returning frames k periods old
FOREVER — the queue never drains, because the consumer runs at exactly the
camera rate. That is a latency LOCK, and no per-frame timing measurement shows
it: every stage looks fast while the picture is a fifth of a second behind.

libcamera stamps each request with SensorTimestamp (ns, CLOCK_BOOTTIME), so
frame age = now - SensorTimestamp is directly measurable.

Modes:
  --mode raw      just capture_request() in a tight loop (baseline camera age)
  --mode pipeline capture + make_array + decode, i.e. the real capture thread
  --mode loaded   as pipeline, plus a synthetic render load per frame

    .venv/bin/python var/hit/frame_latency.py --mode pipeline --seconds 25
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.main import _build_camera


def boottime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))] if s else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--mode", choices=["raw", "pipeline", "loaded"], default="pipeline")
    ap.add_argument("--load-ms", type=float, default=25.0,
                    help="synthetic per-frame work for --mode loaded")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    cam = _build_camera(cfg)
    cam.open()
    picam = cam._picam

    ages = []
    age_by_index = []
    t_end = time.monotonic() + args.seconds
    i = 0
    while time.monotonic() < t_end:
        req = picam.capture_request()
        try:
            md = req.get_metadata()
            if args.mode != "raw":
                frame = req.make_array("main")
        finally:
            req.release()
        ts = md.get("SensorTimestamp")
        if ts is None:
            print("no SensorTimestamp in metadata; keys:", sorted(md)[:20])
            break
        age_ms = (boottime_ns() - int(ts)) / 1e6
        ages.append(age_ms)
        age_by_index.append((i, age_ms))
        if args.mode != "raw":
            cam._decode_detections(md, frame.shape[1], frame.shape[0])
        if args.mode == "loaded":
            t0 = time.monotonic()
            while (time.monotonic() - t0) * 1e3 < args.load_ms:
                pass
        i += 1
    cam.close()

    if not ages:
        return 1
    period = 1000.0 / cfg.camera.framerate
    print()
    print(f"=== frame age, mode={args.mode} ({len(ages)} frames, "
          f"{len(ages)/args.seconds:.1f} fps) ===")
    print(f"  age   med {statistics.median(ages):6.1f}ms   p90 {pct(ages,90):6.1f}ms   "
          f"max {max(ages):6.1f}ms   min {min(ages):6.1f}ms")
    print(f"  frame period at framerate {cfg.camera.framerate}: {period:.1f}ms  "
          f"-> median age = {statistics.median(ages)/period:.2f} frame periods of queue")
    # Drift: is the age growing (queue filling and staying full)?
    first = [a for _, a in age_by_index[: len(ages) // 4]]
    last = [a for _, a in age_by_index[-len(ages) // 4:]]
    print(f"  first quarter med {statistics.median(first):6.1f}ms   "
          f"last quarter med {statistics.median(last):6.1f}ms   "
          f"drift {statistics.median(last)-statistics.median(first):+6.1f}ms")
    if statistics.median(last) - statistics.median(first) > period * 0.75:
        print("  ^^ AGE IS GROWING: completed requests are queueing up (latency lock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
