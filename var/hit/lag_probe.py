"""Measure where the displayed-box latency actually comes from, on the airframe.

Runs the real IMX500 camera + tracker + filter through the real Pipeline with a
null FC and a null display sink, and reports:

  capture      frames/s off the sensor, and the frame-interval spread
  detection    how many captured frames carry a FRESH detection tensor, and the
               interval between fresh ones (the true detection rate)
  control      control-tick rate + tick duration (the guidance loop)
  overlay age  age of the control state that a render would draw on the frame it
               is drawing (this IS the visible box lag)

Usage (on the Pi, service stopped):
    .venv/bin/python var/hit/lag_probe.py --seconds 30
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.fc.null import NullFC
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.main import _build_camera, _build_tracker


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--inject", action="store_true",
                    help="overwrite the sensor's detections with a synthetic target "
                         "moving at 200 px/s, so overlay age is measurable on an empty bench")
    ap.add_argument("--real-fc", action="store_true",
                    help="use the configured ArduPilot backend (includes the MAVLink "
                         "drain in the tick cost) instead of NullFC. STANDBY only.")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    camera = _build_camera(cfg)
    tracker = _build_tracker(cfg)
    if args.real_fc:
        from pi_fpv_companion.main import _build_fc
        fc = _build_fc(cfg)
        fc.open()
    else:
        fc = NullFC()

    cap_ts = []          # monotonic time each frame was captured
    det_fresh_ts = []    # capture time of frames carrying a NEW detection signature
    tick_dur = []        # control tick wall duration
    tick_ts = []         # control tick start times
    overlay_age = []     # age of the control state at render time
    n_render = [0]

    last_sig = [None]

    def display(target, intent, gated, switch, armed, frame, tracks=None):
        # Stands in for the framebuffer render. The state handed to us was produced
        # by the control tick from an EARLIER frame; measure how much earlier.
        n_render[0] += 1
        if target is not None:
            overlay_age.append(frame.timestamp - target.timestamp)

    pipe = Pipeline(
        camera, tracker, cfg.servo, cfg.safety, fc,
        detector=None,
        detect_period_frames=cfg.detector.detect_period_frames,
        display=display,
        camera_watchdog_s=0.0,
        first_frame_grace_s=cfg.camera.first_frame_grace_s,
        rate_cfg=None,          # attitude path; we only care about timing here
    )

    orig_tick = pipe.tick

    def timed_tick(bundle):
        t0 = time.monotonic()
        tick_ts.append(t0)
        try:
            return orig_tick(bundle)
        finally:
            tick_dur.append(time.monotonic() - t0)
    pipe.tick = timed_tick

    orig_frames = camera.frames

    def counted_frames():
        from dataclasses import replace as _replace
        from pi_fpv_companion.types import Detection
        t0 = None
        for b in orig_frames():
            cap_ts.append(time.monotonic())
            if args.inject:
                # A target sweeping horizontally at 200 px/s — fast enough that a
                # stale overlay is obvious, slow enough to stay associated.
                t0 = t0 if t0 is not None else b.timestamp
                phase = ((b.timestamp - t0) * 200.0) % (b.width - 120.0)
                b = _replace(b, detections=[Detection(
                    x=60.0 + phase, y=b.height / 2.0, w=80.0, h=80.0,
                    confidence=0.9, class_id=0, class_name="person")])
            sig = Pipeline._detection_sig(b.detections)
            if sig != last_sig[0]:
                last_sig[0] = sig
                det_fresh_ts.append(b.timestamp)
            yield b
    camera.frames = counted_frames

    import threading
    threading.Timer(args.seconds, pipe.stop).start()
    t_start = time.monotonic()
    pipe.run()
    elapsed = time.monotonic() - t_start

    def gaps(ts):
        return [b - a for a, b in zip(ts, ts[1:])]

    cap_gaps = gaps(cap_ts)
    det_gaps = gaps(det_fresh_ts)
    ctl_gaps = gaps(tick_ts)

    print()
    print(f"=== lag probe: {elapsed:.1f}s, config {args.config} ===")
    print(f"capture      {len(cap_ts)/elapsed:6.1f} fps   "
          f"gap med {statistics.median(cap_gaps)*1e3 if cap_gaps else 0:5.1f}ms  "
          f"p90 {pct(cap_gaps,90)*1e3:5.1f}ms  max {max(cap_gaps)*1e3 if cap_gaps else 0:6.1f}ms")
    print(f"detection    {len(det_fresh_ts)/elapsed:6.1f} Hz    "
          f"fresh on {100.0*len(det_fresh_ts)/max(1,len(cap_ts)):4.1f}% of frames  "
          f"gap med {statistics.median(det_gaps)*1e3 if det_gaps else 0:5.1f}ms  "
          f"p90 {pct(det_gaps,90)*1e3:5.1f}ms")
    print(f"control      {len(tick_ts)/elapsed:6.1f} Hz    "
          f"tick med {statistics.median(tick_dur)*1e3 if tick_dur else 0:5.1f}ms  "
          f"p90 {pct(tick_dur,90)*1e3:5.1f}ms  max {max(tick_dur)*1e3 if tick_dur else 0:6.1f}ms  "
          f"gap med {statistics.median(ctl_gaps)*1e3 if ctl_gaps else 0:5.1f}ms")
    print(f"render       {n_render[0]/elapsed:6.1f} fps")
    if overlay_age:
        print(f"OVERLAY AGE  med {statistics.median(overlay_age)*1e3:5.1f}ms  "
              f"p90 {pct(overlay_age,90)*1e3:5.1f}ms  max {max(overlay_age)*1e3:6.1f}ms  "
              f"(n={len(overlay_age)})")
        print("             ^ this is how far BEHIND the live image the drawn box is")
    else:
        print("OVERLAY AGE  no target was ever locked (put something in frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
