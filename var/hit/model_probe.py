#!/usr/bin/env python3
"""Load an IMX500 model, time open->first-detection (recovery cost), and dump the
raw NPU output-tensor format (so we know if our SSD decoder works for it).

  python3 model_probe.py /usr/share/imx500-models/<model>.rpk
"""
import sys
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

model = sys.argv[1]
t0 = time.monotonic()
imx500 = IMX500(model)
intr = imx500.network_intrinsics
t_ctor = time.monotonic()

picam = Picamera2(imx500.camera_num)
cfg = picam.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    controls={"FrameRate": 30}, buffer_count=8)
picam.configure(cfg)
picam.start(cfg, show_preview=False)
t_start = time.monotonic()

got = None
t_first = None
for _ in range(500):                       # up to ~15s for the rpk upload + first output
    req = picam.capture_request()
    try:
        md = req.get_metadata()
    finally:
        req.release()
    o = imx500.get_outputs(md, add_batch=True)
    if o is not None:
        got = o
        t_first = time.monotonic()
        break
    time.sleep(0.03)

print("MODEL:", model.split("/")[-1])
print(f"  input size      : {imx500.get_input_size()}")
print(f"  IMX500() ctor   : {t_ctor - t0:.2f}s")
if t_first:
    print(f"  start->1st detect: {t_first - t_start:.2f}s  (rpk upload to sensor)")
    print(f"  TOTAL open->detect: {t_first - t0:.2f}s")
    print(f"  output tensors  : {len(got)}  shapes={[np.array(x).shape for x in got]}")
    labels = getattr(intr, 'labels', None)
    print(f"  label count     : {len(labels) if labels else 'n/a'}")
else:
    print("  NO OUTPUT within timeout")
picam.stop()
picam.close()
