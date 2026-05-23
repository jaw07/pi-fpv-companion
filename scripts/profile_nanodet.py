"""Profile NanoDet-Plus inference on the Mac to project Pi Zero 2W latency.

Doesn't run the full pipeline — just the detector + a representative frame.
Use this to validate that a candidate model file + input size fits the Pi budget
BEFORE wiring it into the demo loop.

Usage:
    .venv/bin/python scripts/profile_nanodet.py \
        --model-dir /Users/user/development/drone-guidance/models/ncnn/nanodet_plus_m_416 \
        --input-size 416 \
        --frames 30
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.detect.nanodet import NanoDetConfig, NanoDetDetector
from pi_fpv_companion.perf import PerfMonitor, PiBudget


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pi-scale", type=float, default=6.0,
                    help="Mac->Pi Zero 2W scaling factor (~6x for NCNN/A53 vs M-series)")
    args = ap.parse_args()

    cfg = NanoDetConfig(
        model_dir=args.model_dir,
        input_size=args.input_size,
        conf_threshold=args.conf,
    )
    print(f"loading model from {args.model_dir} (input={args.input_size}x{args.input_size}) ...")
    det = NanoDetDetector(cfg)
    det.open()
    print("model loaded")

    cam = SyntheticCamera(width=720, height=576)
    sample_frame = cam.render_at(1.5).image
    print(f"sample frame: {sample_frame.shape}")

    print(f"warmup ({args.warmup} iterations) ...")
    for i in range(args.warmup):
        t0 = time.perf_counter()
        det.detect(sample_frame)
        print(f"  warmup {i+1}: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # The pi budget here describes the *detector* alone — 200 ms is what we allotted
    # per call in the PiCam path config, leaving the rest for tracker/IO/overlay.
    perf = PerfMonitor(
        PiBudget(max_tick_ms=200.0, max_rss_mb=200.0, pi_scale_factor=args.pi_scale)
    )

    print(f"timing {args.frames} detections ...")
    detection_counts = []
    for _ in range(args.frames):
        t0 = perf.tick_start()
        dets = det.detect(sample_frame)
        perf.tick_end(t0)
        detection_counts.append(len(dets))

    print()
    print(perf.report())
    print()
    print(f"detections per frame: min={min(detection_counts)}  max={max(detection_counts)}  "
          f"mean={np.mean(detection_counts):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
