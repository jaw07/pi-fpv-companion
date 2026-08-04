"""Measure the IMX500's TRUE inference refresh rate, independent of conf threshold.

The config asserts that at framerate 22 "EVERY frame carries a fresh detection".
That claim is about the on-sensor NPU tensor, not about boxes that pass
conf_threshold — so test it by hashing the raw output tensor per frame and
counting how often it CHANGES. Also reports how many frames carry any box at
several confidence levels, and dumps a frame so we can see the scene.

    .venv/bin/python var/hit/tensor_rate.py --seconds 20
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.main import _build_camera


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--dump", default="var/hit/scene.jpg")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    cam = _build_camera(cfg)
    cam.open()

    imx = cam._imx500
    n_frames = 0
    fresh_ts = []          # capture time of frames whose NPU tensor CHANGED
    frame_ts = []
    last_h = None
    best_scores = []
    saved = False
    t_end = time.monotonic() + args.seconds

    picam = cam._picam
    while time.monotonic() < t_end:
        req = picam.capture_request()
        try:
            md = req.get_metadata()
            if not saved:
                import cv2
                cv2.imwrite(args.dump, req.make_array("main"))
                saved = True
        finally:
            req.release()
        now = time.monotonic()
        n_frames += 1
        frame_ts.append(now)
        outs = imx.get_outputs(md, add_batch=True)
        if outs is None or len(outs) < 4:
            continue
        h = hash(np.asarray(outs[0]).tobytes() + np.asarray(outs[1]).tobytes())
        if h != last_h:
            last_h = h
            fresh_ts.append(now)
        scores = np.asarray(outs[1]).reshape(-1)
        if scores.size:
            best_scores.append(float(scores.max()))

    cam.close()
    elapsed = frame_ts[-1] - frame_ts[0] if len(frame_ts) > 1 else 1.0
    gaps = [b - a for a, b in zip(fresh_ts, fresh_ts[1:])]
    print()
    print(f"=== IMX500 tensor rate ({elapsed:.1f}s, framerate cfg={cfg.camera.framerate}) ===")
    print(f"frames captured   {n_frames}  ->  {n_frames/elapsed:.1f} fps")
    print(f"tensor CHANGED    {len(fresh_ts)}  ->  {len(fresh_ts)/elapsed:.1f} Hz  "
          f"({100.0*len(fresh_ts)/max(1,n_frames):.0f}% of frames)")
    if gaps:
        print(f"  refresh gap     med {statistics.median(gaps)*1e3:.1f}ms  "
              f"min {min(gaps)*1e3:.1f}ms  max {max(gaps)*1e3:.1f}ms")
    if best_scores:
        arr = np.array(best_scores)
        print(f"top score/frame   med {np.median(arr):.3f}  max {arr.max():.3f}")
        for thr in (0.20, 0.35, 0.50):
            print(f"  frames with a box >= {thr:.2f}: "
                  f"{100.0*(arr >= thr).mean():.0f}%")
    print(f"scene dumped to   {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
