"""Reproduce the STANDBY jitter/lag with a realistically NOISY, near-stationary target.

The moving-target probe showed a clean 22 Hz control loop and a 45 ms box lag.
But a bench/STANDBY target is usually near-stationary, and a real detector emits
sub-pixel-noisy boxes around it. Pipeline._detection_sig() rounds boxes to whole
pixels and SKIPS the control tick when the signature is unchanged, so the loop
rate becomes a function of how much the detector happens to be dithering.

This injects a target with a configurable jitter amplitude and reports the
control rate and, most importantly, how long the drawn box FREEZES between
updates while the video keeps running at 22 fps.

    .venv/bin/python var/hit/standby_jitter.py --seconds 20 --jitter 0.0
    .venv/bin/python var/hit/standby_jitter.py --seconds 20 --jitter 0.4
    .venv/bin/python var/hit/standby_jitter.py --seconds 20 --jitter 2.0
"""
from __future__ import annotations
import argparse
import math
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
from pi_fpv_companion.main import _build_camera, _build_tracker
from pi_fpv_companion.fc.null import NullFC


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))] if s else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--jitter", type=float, default=0.4,
                    help="px amplitude of detector noise on a stationary target")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="px/s of slow real motion")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    camera = _build_camera(cfg)
    tracker = _build_tracker(cfg)

    tick_ts, frame_ts = [], []
    box_x, box_ts = [], []
    truth_err = []
    t_origin = []            # capture-side t0, for the ground-truth position

    def display(target, intent, gated, switch, armed, frame, tracks=None):
        frame_ts.append(frame.timestamp)
        if target is not None:
            box_x.append(target.detection.x)
            box_ts.append(frame.timestamp)
            if t_origin:
                # Where the injected target ACTUALLY is at this frame's timestamp.
                truth = 360.0 + args.drift * (frame.timestamp - t_origin[0])
                truth_err.append(target.detection.x - truth)

    pipe = Pipeline(
        camera, tracker, cfg.servo, cfg.safety, NullFC(),
        detector=None, display=display,
        camera_watchdog_s=0.0,
        first_frame_grace_s=cfg.camera.first_frame_grace_s,
    )
    orig_tick = pipe.tick

    def timed_tick(b):
        tick_ts.append(time.monotonic())
        return orig_tick(b)
    pipe.tick = timed_tick

    orig_frames = camera.frames

    def injected():
        t0 = None
        n = 0
        for b in orig_frames():
            if t0 is None:
                t0 = b.timestamp
                t_origin.append(t0)
            dt = b.timestamp - t0
            # deterministic pseudo-noise, so runs are comparable
            noise = args.jitter * math.sin(n * 2.399963)
            n += 1
            yield _replace(b, detections=[Detection(
                x=360.0 + args.drift * dt + noise, y=288.0,
                w=80.0, h=80.0, confidence=0.9, class_id=0, class_name="person")])
    camera.frames = injected

    threading.Timer(args.seconds, pipe.stop).start()
    t0 = time.monotonic()
    pipe.run()
    elapsed = time.monotonic() - t0

    # How long does the DRAWN box sit at the same value while frames keep coming?
    freezes = []
    run_start = None
    for i in range(1, len(box_x)):
        if abs(box_x[i] - box_x[i - 1]) < 1e-9:
            run_start = run_start if run_start is not None else box_ts[i - 1]
        else:
            if run_start is not None:
                freezes.append(box_ts[i] - run_start)
                run_start = None

    ctl_gaps = [b - a for a, b in zip(tick_ts, tick_ts[1:])]
    print()
    print(f"=== STANDBY jitter probe: noise={args.jitter}px drift={args.drift}px/s "
          f"({elapsed:.1f}s) ===")
    print(f"  frames displayed   {len(frame_ts)/elapsed:5.1f} fps")
    print(f"  control ticks      {len(tick_ts)/elapsed:5.1f} Hz   "
          f"gap med {statistics.median(ctl_gaps)*1e3 if ctl_gaps else 0:6.1f}ms  "
          f"max {max(ctl_gaps)*1e3 if ctl_gaps else 0:6.1f}ms")
    if freezes:
        print(f"  box FROZE          {len(freezes)} times  "
              f"med {statistics.median(freezes)*1e3:6.1f}ms  "
              f"p90 {pct(freezes,90)*1e3:6.1f}ms  max {max(freezes)*1e3:6.1f}ms")
        print(f"  -> the drawn box is stale for {100.0*sum(freezes)/elapsed:.0f}% "
              f"of the time the video is live")
    else:
        print("  box updated on every displayed frame (no freeze)")
    if truth_err and args.drift:
        import statistics as st
        print(f"  box vs TRUTH       mean {st.mean(truth_err):+6.1f}px  "
              f"med {st.median(truth_err):+6.1f}px  "
              f"|err| med {st.median([abs(e) for e in truth_err]):5.1f}px")
        print(f"     (a box lagging by one 45ms frame at {args.drift:.0f}px/s "
              f"would sit {args.drift*0.0453:.1f}px behind)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
