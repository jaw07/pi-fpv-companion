#!/usr/bin/env python3
"""Bench test A: live IMX500 detection + tracking probe.

Reuses the real flight components (IMX500Camera + tracker, same imx500.yaml
thresholds/classes) and prints what the on-sensor CNN sees plus whether the
tracker locks. No FC connection, no motor commands — read-only perception test.

  .venv/bin/python var/hit/detect_probe.py [seconds]
"""
import sys
import time
from collections import Counter

from pi_fpv_companion.config import load
from pi_fpv_companion.main import _build_camera, _build_tracker

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
# Optional bench overrides: arg2 = conf threshold, arg3 = "any" to drop the
# class filter (lets the tracker lock onto whatever's in view as a stand-in
# target when no flight-class object is present).
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else None
ANYCLASS = len(sys.argv) > 3 and sys.argv[3].lower().startswith("any")

cfg = load("config/imx500.yaml")
if CONF is not None:
    cfg.detector.conf_threshold = CONF
if ANYCLASS:
    cfg.detector.classes_of_interest = []   # empty -> camera accepts any class
print(f"conf_threshold={cfg.detector.conf_threshold} "
      f"classes={cfg.detector.classes_of_interest or 'ANY'}")
cam = _build_camera(cfg)
trk = _build_tracker(cfg)
# STANDBY auto-acquire: lock the best current detection without an RC select.
if hasattr(trk, "auto_acquire"):
    trk.auto_acquire = True

cam.open()
print("camera open — show it a person / car / etc. Watching for "
      f"{SECS:.0f}s ...\n")

frames = 0
frames_with_det = 0
max_dets = 0
classes_seen = Counter()
best_conf = 0.0
ever_locked = False
locked_frames = 0
last_print = 0.0
t0 = time.monotonic()

try:
    for b in cam.frames():
        frames += 1
        dets = b.detections
        if dets:
            frames_with_det += 1
            max_dets = max(max_dets, len(dets))
            for d in dets:
                classes_seen[d.class_name or d.class_id] += 1
                best_conf = max(best_conf, d.confidence)

        tgt = trk.consume(b.image, dets, b.timestamp)
        locked = bool(getattr(trk, "is_locked", lambda: tgt is not None)())
        if locked:
            ever_locked = True
            locked_frames += 1

        now = time.monotonic()
        if now - last_print >= 0.5:
            last_print = now
            if dets:
                top = max(dets, key=lambda d: d.confidence)
                summary = ", ".join(
                    f"{(d.class_name or d.class_id)}:{d.confidence:.2f}" for d in dets[:4])
                lock = (f"LOCK id={tgt.track_id} lost={tgt.lost_frames} "
                        f"@({tgt.detection.x:.0f},{tgt.detection.y:.0f})"
                        if tgt else "no-lock")
                print(f"[{now-t0:4.1f}s] f{frames:4d}  {len(dets)} det  "
                      f"[{summary}]  top@({top.x:.0f},{top.y:.0f}) "
                      f"{top.w:.0f}x{top.h:.0f}px  {lock}")
            else:
                print(f"[{now-t0:4.1f}s] f{frames:4d}  no detections")

        if now - t0 >= SECS:
            break
finally:
    cam.close()

print("\n===== detection probe summary =====")
print(f"frames                {frames}  ({frames/max(now-t0,1e-9):.1f} fps)")
print(f"frames with a det     {frames_with_det}  "
      f"({100*frames_with_det/max(frames,1):.0f}%)")
print(f"max dets in a frame   {max_dets}")
print(f"best confidence       {best_conf:.2f}")
print(f"classes seen          {dict(classes_seen)}")
print(f"tracker ever locked   {ever_locked}  "
      f"(locked {locked_frames}/{frames} frames)")
if frames_with_det == 0:
    print("\nNO DETECTIONS — check the subject is one of the classes, well-lit, "
          "and filling enough of the frame; conf_threshold may be too high.")
elif not ever_locked:
    print("\nDetections seen but tracker never locked — investigate tracker "
          "association.")
else:
    print("\nPERCEPTION OK: on-sensor CNN detected + tracker locked.")
