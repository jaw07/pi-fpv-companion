"""Compare AE exposure profiles on the real sensor — RUN THIS IN DAYLIGHT.

Exposure time is the motion-blur budget: on a moving airframe a long exposure smears
the target and the network stops finding it. The `short` AE profile trades exposure
for gain ~3x sooner than `normal` (see config/imx500.yaml), but the benefit only shows
where AE has headroom to give back. On a dark bench both profiles peg at the frame
duration and this probe will (correctly) report no difference.

Reports, per profile: exposure time, analogue gain, scene lux, and the implied image
smear for a given airframe rotation rate.

    .venv/bin/python var/hit/exposure_probe.py --deg-per-s 60
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.main import _build_camera

HFOV_DEG = 66.0     # IMX500 horizontal FoV (vfov_deg 52.3 at 720x576 -> ~66 h)


def sample(cfg, mode: str, n: int, width: int):
    cfg.camera.ae_exposure_mode = mode
    cam = _build_camera(cfg)
    cam.open()
    picam = cam._picam
    exp, gain, lux = [], [], []
    for i in range(n):
        req = picam.capture_request()
        try:
            md = req.get_metadata()
        finally:
            req.release()
        if i < n // 3:
            continue                      # let AE settle before recording
        exp.append(md.get("ExposureTime", 0))
        gain.append(md.get("AnalogueGain", 0.0))
        lux.append(md.get("Lux", 0.0))
    cam.close()
    return exp, gain, lux


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/imx500.yaml")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--deg-per-s", type=float, default=60.0,
                    help="airframe rotation rate to compute image smear for")
    args = ap.parse_args()

    cfg = load(Path(args.config))
    width = cfg.video.width
    px_per_deg = width / HFOV_DEG

    print(f"frame {width}px over {HFOV_DEG:.0f} deg -> {px_per_deg:.1f} px/deg; "
          f"smear quoted at {args.deg_per_s:.0f} deg/s of airframe rotation")
    print()
    print(f"{'profile':<10} {'exposure':>12} {'gain':>7} {'lux':>8} {'smear':>10}")
    for mode in ("normal", "short"):
        exp, gain, lux = sample(cfg, mode, args.frames, width)
        if not exp:
            continue
        e = statistics.median(exp)
        smear_px = (e / 1e6) * args.deg_per_s * px_per_deg
        print(f"{mode:<10} {e/1000.0:9.1f}ms {statistics.median(gain):7.1f} "
              f"{statistics.median(lux):8.1f} {smear_px:8.1f}px")
        time.sleep(1.0)
    print()
    print("If both rows are identical the scene is too dark for AE to have any choice —")
    print("re-run outdoors. A target only a few px wide cannot survive smear of its own size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
