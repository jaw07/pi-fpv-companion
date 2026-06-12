#!/usr/bin/env python3
"""Stress the IMX500 open->detect->close cycle to characterise the capture hang and
recovery timing. Each cycle mimics what a watchdog restart does (minus the process
respawn). Reports per-cycle open->first-detection time and flags any hang/failure.

  python3 recovery_stress.py [cycles]
"""
import sys
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
HANG_S = 15.0                      # treat >this with no first frame as a hang

times = []
hangs = 0
for c in range(1, CYCLES + 1):
    t0 = time.monotonic()
    imx500 = IMX500(MODEL)
    picam = Picamera2(imx500.camera_num)
    cfg = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        controls={"FrameRate": 30}, buffer_count=8)
    picam.configure(cfg)
    picam.start(cfg, show_preview=False)
    first = None
    while time.monotonic() - t0 < HANG_S:
        req = picam.capture_request()
        try:
            arr = req.make_array("main")          # also exercises the frame path
        finally:
            req.release()
        if arr is not None:
            first = time.monotonic()
            break
    dt = (first - t0) if first else None
    if dt is None:
        hangs += 1
        print(f"  cycle {c:2d}: HANG (no frame in {HANG_S:.0f}s)")
    else:
        times.append(dt)
        print(f"  cycle {c:2d}: open->first-frame {dt:.2f}s")
    picam.stop()
    picam.close()
    time.sleep(0.5)

print("\n=== recovery stress summary ===")
print(f"cycles={CYCLES}  hangs={hangs}")
if times:
    t = np.array(times)
    print(f"open->first-frame  min={t.min():.2f}s  med={np.median(t):.2f}s  max={t.max():.2f}s")
