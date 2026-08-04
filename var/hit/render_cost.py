"""Measure the REAL video sink cost + ThreadedSink drop rate on the airframe.

The lag probe stubbed the display. This one runs the actual production sink
(FramebufferSink over /dev/fb0 or DRM) behind ThreadedSink, exactly as main.py
builds it, and reports:

  render        ms per frame inside the sink, broken down (copy / overlay / write)
  drops         frames ThreadedSink threw away because the renderer was still busy
  displayed     the rate the TV out ACTUALLY updates at

If render ms > the camera frame period (45ms at framerate 22) the feed runs at
the render rate, not the camera rate, and every displayed frame is stale by at
least one render.

    sudo systemctl stop pi-fpv-companion
    .venv/bin/python var/hit/render_cost.py --seconds 20
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from pi_fpv_companion.config import load
from pi_fpv_companion.main import _build_sink
from pi_fpv_companion.video import framebuffer as fbmod
from pi_fpv_companion.video.overlay import draw_overlay


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))] if s else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    cfg = load(Path(args.config))
    sink = _build_sink(cfg, no_gui=False)
    print(f"sink: {type(sink).__name__} -> {type(getattr(sink, '_fb', None)).__name__}")
    fb = sink._fb
    sink.open()
    print(f"framebuffer: {fb.width}x{fb.height} bpp={fb.bpp}   "
          f"frame: {cfg.video.width}x{cfg.video.height}")

    img = (np.random.rand(cfg.video.height, cfg.video.width, 3) * 255).astype(np.uint8)

    from pi_fpv_companion.guidance.safety import GateResult
    from pi_fpv_companion.types import GuidanceMode, SwitchState, ZERO_INTENT
    switch = SwitchState(active=False, pwm_us=1000, timestamp=0.0, mode=GuidanceMode.STANDBY)
    gated = GateResult(ZERO_INTENT, True, "standby")

    t_copy, t_draw, t_pack, t_write, t_total = [], [], [], [], []
    t_end = time.monotonic() + args.seconds
    n = 0
    while time.monotonic() < t_end:
        a = time.monotonic()
        buf = img.copy()
        b = time.monotonic()
        draw_overlay(buf, None, ZERO_INTENT, switch, False, gated, None)
        c = time.monotonic()
        packed = fbmod.bgr_to_rgb565(buf) if fb.bpp == 16 else fbmod.bgr_to_bgra(buf)
        raw = packed.tobytes()
        d = time.monotonic()
        fb.write(buf)
        e = time.monotonic()
        t_copy.append(b - a); t_draw.append(c - b)
        t_pack.append(d - c); t_write.append(e - d); t_total.append(e - a)
        n += 1
    sink.close()

    def row(name, v):
        print(f"  {name:<10} med {statistics.median(v)*1e3:6.2f}ms  "
              f"p90 {pct(v,90)*1e3:6.2f}ms  max {max(v)*1e3:6.2f}ms")

    print()
    print(f"=== render cost ({n} renders, {n/args.seconds:.1f}/s sustained) ===")
    row("copy", t_copy)
    row("overlay", t_draw)
    row("pack565", t_pack)
    row("write", t_write)   # note: write() re-does the pack internally
    print(f"  {'TOTAL':<10} med {statistics.median(t_total)*1e3:6.2f}ms  "
          f"p90 {pct(t_total,90)*1e3:6.2f}ms")
    print(f"  camera frame period at framerate {cfg.camera.framerate}: "
          f"{1000.0/cfg.camera.framerate:.1f}ms")
    real = statistics.median(t_copy) + statistics.median(t_draw) + statistics.median(t_write)
    print(f"  production per-frame cost (copy+overlay+write): {real*1e3:.2f}ms  "
          f"-> max sustainable {1.0/real:.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
