#!/usr/bin/env python3
"""Grab one frame as JPEG + dump raw IMX500 NPU outputs (threshold ~0, all classes).

Tells us (a) what the camera is actually pointed at, and (b) whether the on-sensor
detection decode path produces ANY candidates on the current scene — independent of
whether a target-of-interest is present.

  .venv/bin/python var/hit/frame_and_raw.py
"""
import time
import numpy as np
import cv2

from pi_fpv_companion.config import load
from pi_fpv_companion.camera.imx500 import IMX500Camera
from pi_fpv_companion.detect.coco import COCO_CLASSES

cfg = load("config/imx500.yaml")
# No class filter, threshold ~0 so we see the raw candidate tail.
cam = IMX500Camera(
    model_path=cfg.camera.imx500_model,
    width=cfg.video.width, height=cfg.video.height,
    framerate=cfg.camera.framerate,
    conf_threshold=0.0,
    target_class_ids=(),
)
cam.open()
labels = cam._labels

# Let AE/AWB settle (~1s), then capture one request for frame + raw metadata.
picam = cam._picam
time.sleep(1.0)
for _ in range(30):          # discard warm-up frames
    picam.capture_request().release()
req = picam.capture_request()
try:
    frame = req.make_array("main")
    metadata = req.get_metadata()
finally:
    req.release()

cv2.imwrite("var/hit/frame.jpg", frame)
print(f"saved var/hit/frame.jpg  ({frame.shape[1]}x{frame.shape[0]})")
print(f"mean pixel {frame.mean():.1f}  (near 0 = black/lens cap, near 255 = blown out)")

outputs = cam._imx500.get_outputs(metadata, add_batch=True)
if outputs is None:
    print("get_outputs returned None — NO NPU output in metadata this frame")
else:
    print(f"raw NPU outputs: {len(outputs)} tensors, shapes "
          f"{[np.array(o).shape for o in outputs]}")
    try:
        scores = np.array(outputs[1]).reshape(-1)
        classes = np.array(outputs[2]).reshape(-1).astype(int)
        n = int(np.array(outputs[3]).reshape(-1)[0])
        order = np.argsort(-scores)[:8]
        print(f"count tensor = {n}; top-8 candidates (any class):")
        for j in order:
            cname = labels[classes[j]] if 0 <= classes[j] < len(labels) else "?"
            print(f"   score={scores[j]:.3f}  class={classes[j]} ({cname})")
    except Exception as e:
        print(f"could not parse score/class tensors: {e}")

cam.close()
