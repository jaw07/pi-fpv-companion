"""End-to-end glass-to-framebuffer latency through the REAL production stack.

Everything measured so far was a stage in isolation and every stage looked fast.
This runs exactly what the service runs — IMX500 -> Pipeline (decoupled) ->
ThreadedSink -> FramebufferSink -> /dev/fb0 — and instruments the moment the
pixels actually hit the framebuffer:

  bundle age   now - bundle.timestamp at fb-write time (pipeline-side latency)
  total age    bundle age + the measured sensor->userspace age (add --sensor-ms)
  display fps  how often the framebuffer is actually updated
  drops        frames ThreadedSink discarded

Injects a synthetic moving target (like lag_probe --inject) so the tracker,
filter, guidance and overlay all run on a real, moving lock.

    .venv/bin/python var/hit/fullstack_lag.py --seconds 25 --sensor-ms 72.5
"""
from __future__ import annotations
import argparse
import statistics
import sys
import threading
import time
from dataclasses import replace as _replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.types import Detection
from pi_fpv_companion.main import _build_camera, _build_tracker, _build_fc, _build_sink
from pi_fpv_companion.video.threaded_sink import ThreadedSink


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))] if s else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--sensor-ms", type=float, default=72.5,
                    help="measured sensor->userspace age (frame_latency.py)")
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    camera = _build_camera(cfg)
    tracker = _build_tracker(cfg)
    fc = _build_fc(cfg)
    fc.open()
    raw_sink = _build_sink(cfg, no_gui=False)

    bundle_age = []      # now - bundle.timestamp, at the instant of fb write
    box_lag = []         # how stale the DRAWN BOX is vs the frame it's drawn on
    disp_ts = []

    orig_show = raw_sink.show

    def timed_show(target, intent, gated, switch, armed, frame, tracks=None):
        orig_show(target, intent, gated, switch, armed, frame, tracks)
        now = time.monotonic()
        disp_ts.append(now)
        bundle_age.append(now - frame.timestamp)
        if target is not None:
            box_lag.append(frame.timestamp - target.timestamp)

    raw_sink.show = timed_show
    sink = ThreadedSink(raw_sink)

    pipe = Pipeline(
        camera, tracker, cfg.servo, cfg.safety, fc,
        detector=None,
        detect_period_frames=cfg.detector.detect_period_frames,
        display=sink.show,
        camera_watchdog_s=0.0,
        first_frame_grace_s=cfg.camera.first_frame_grace_s,
        rate_cfg=None,
    )

    if not args.no_inject:
        orig_frames = camera.frames

        def injected():
            t0 = None
            for b in orig_frames():
                t0 = t0 if t0 is not None else b.timestamp
                phase = ((b.timestamp - t0) * 200.0) % (b.width - 120.0)
                yield _replace(b, detections=[Detection(
                    x=60.0 + phase, y=b.height / 2.0, w=80.0, h=80.0,
                    confidence=0.9, class_id=0, class_name="person")])
        camera.frames = injected

    threading.Timer(args.seconds, pipe.stop).start()
    t0 = time.monotonic()
    pipe.run()
    elapsed = time.monotonic() - t0
    sink.close()
    fc.close()

    rendered, dropped = sink.stats
    print()
    print(f"=== full-stack ({elapsed:.1f}s) ===")
    print(f"displayed        {len(disp_ts)/elapsed:5.1f} fps to /dev/fb0")
    print(f"ThreadedSink     rendered {rendered}  dropped {dropped} "
          f"({100.0*dropped/max(1,rendered+dropped):.0f}%)")
    if bundle_age:
        print(f"pipeline age     med {statistics.median(bundle_age)*1e3:6.1f}ms  "
              f"p90 {pct(bundle_age,90)*1e3:6.1f}ms  max {max(bundle_age)*1e3:6.1f}ms")
        print(f"  + sensor->userspace {args.sensor_ms:.1f}ms")
        print(f"  = GLASS-TO-FRAMEBUFFER  med "
              f"{statistics.median(bundle_age)*1e3 + args.sensor_ms:6.1f}ms")
    if box_lag:
        print(f"box vs image     med {statistics.median(box_lag)*1e3:6.1f}ms  "
              f"p90 {pct(box_lag,90)*1e3:6.1f}ms  max {max(box_lag)*1e3:6.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
