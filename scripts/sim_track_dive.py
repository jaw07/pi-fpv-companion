"""FOV-retention + convergence envelope for TRACK and DIVE (no hardware).

Flies the closed-loop simulator (tests/closed_loop_sim.py — the REAL filter,
servo and safety gate against a fixed, airframe-bolted pinhole camera) across a
grid of geometries and prints where the guidance KEEPS the target in frame and
converges versus where it loses it. This is the question single-frame unit tests
cannot answer: the camera rotates with every command, so the loop can steer its
own field of view off the target.

Defaults model the Raspberry Pi AI Camera (Sony IMX500): HFoV 66.3, VFoV 52.3.

    .venv/bin/python scripts/sim_track_dive.py
    .venv/bin/python scripts/sim_track_dive.py --vfov 40   # narrower lens stress

Outcomes:  HIT = closed < impact range, in frame throughout
           blind = target outside the FoV at acquisition (lens limit, not guidance)
           LOST = left the frame mid-engagement
           miss<n> = stayed framed but did not close (closest approach n m)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from tests.closed_loop_sim import (   # noqa: E402
    Airframe, CameraModel, SimWorld, imx500_servo, imx500_safety,
)
from pi_fpv_companion.types import GuidanceMode   # noqa: E402

W, H = 720, 576


def outcome(tr) -> str:
    if not any(tk.in_frame for tk in tr.ticks):
        return "blind"
    if tr.lost_before_impact(5.0):     # terminal frame-exit at impact is not a loss
        return "LOST"
    if tr.min_range < 5.0:
        return "HIT"
    return f"miss{tr.min_range:.0f}"


def world(target_pos, *, vfov, alt=50.0, target_vel=(0.0, 0.0, 0.0), **servo):
    cam = CameraModel(W, H, vfov_deg=vfov)
    return SimWorld(camera=cam, servo=imx500_servo(**servo), safety=imx500_safety(),
                    airframe=Airframe(pos=(0.0, 0.0, alt)),
                    target_pos=target_pos, target_vel=target_vel)


def hdr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def track_crossing_envelope(vfov):
    hdr("TRACK — crossing-target FOV retention (does yaw keep the FOV on it?)")
    print("rows = lateral target speed (m/s), cols = initial range (m). alt-matched.")
    ranges = [8, 12, 20, 30, 45]
    print(f"  {'speed':>7} | " + " ".join(f"{r:>7}" for r in ranges))
    for spd in [0, 2, 5, 10, 18, 28]:
        cells = []
        for r in ranges:
            tr = world((r, 0.0, 50.0), vfov=vfov, target_vel=(0.0, spd, 0.0)).run(
                GuidanceMode.TRACK, duration_s=12.0)
            framed = "kept" if not tr.ever_left_frame else "LOST"
            cells.append(f"{framed:>7}")
        print(f"  {spd:>7} | " + " ".join(cells))


def dive_ground_envelope(vfov):
    hdr("DIVE — ground-target envelope: closed-loop homing OFF vs ON")
    print("rows = config × engagement altitude, cols = ground range (m). 'blind' =")
    print("depression at acquisition exceeds half the VFoV (the fixed-camera cone).")
    ranges = [50, 70, 90, 110, 140, 180]
    print(f"  {'config @ alt':<26} | " + " ".join(f"{r:>6}" for r in ranges))
    off = dict(dive_vrate_gain=0.0)   # no vertical homing -> just leans, pancakes
    rows = [
        ("homing OFF @35 m", 35.0, off),
        ("homing ON  @35 m", 35.0, dict()),
        ("homing ON  @50 m", 50.0, dict()),
    ]
    for name, alt, sv in rows:
        cells = []
        for r in ranges:
            tr = world((r, 0.0, 0.0), vfov=vfov, alt=alt, **sv).run(
                GuidanceMode.DIVE, duration_s=120.0)
            cells.append(f"{outcome(tr):>6}")
        print(f"  {name:<26} | " + " ".join(cells))


def dive_altitude_geometries(vfov):
    hdr("DIVE — altitude-agnostic closed-loop homing (below / level / above)")
    cases = [
        ("BELOW  ground  110 m  (alt 50)", (110.0, 0.0, 0.0), 50.0),
        ("BELOW  ground   85 m  +offset", (85.0, -15.0, 0.0), 35.0),
        ("LEVEL  front    50 m, same alt", (50.0, 0.0, 50.0), 50.0),
        ("LEVEL  front    80 m, same alt", (80.0, 0.0, 50.0), 50.0),
        ("ABOVE  +15 m,   55 m ahead", (55.0, 0.0, 65.0), 50.0),
        ("ABOVE  +25 m,  100 m ahead", (100.0, 0.0, 75.0), 50.0),
    ]
    print(f"  {'geometry':<32} {'outcome':>8} {'min_rng':>8} {'alt_d':>7}")
    for name, tp, alt in cases:
        tr = world(tp, vfov=vfov, alt=alt).run(GuidanceMode.DIVE, duration_s=120.0)
        ad = -tr.altitude_lost   # +climb, -descend
        print(f"  {name:<32} {outcome(tr):>8} {tr.min_range:>8.1f} {ad:>+7.1f}")
    print("  (alt_d: + = climbed toward an above target, - = descended onto a below one)")
    print("  Closed-loop homing closes onto a target below, level, OR above — the")
    print("  commanded vertical rate holds the framing so the path follows the LOS.")


def vfov_sensitivity():
    hdr("FOV sensitivity — DIVE on a 110 m ground target across lens VFoV")
    print("  Narrower VFoV raises the start-depression that is acquirable.")
    print(f"  {'VFoV(deg)':>9} {'outcome':>8}   note")
    for vf in [40.0, 45.0, 52.3, 60.0, 70.0]:
        tr = world((110.0, 0.0, 0.0), vfov=vf).run(GuidanceMode.DIVE, duration_s=120.0)
        note = "IMX500 spec" if abs(vf - 52.3) < 0.1 else ""
        print(f"  {vf:>9.1f} {outcome(tr):>8}   {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vfov", type=float, default=52.3, help="camera vertical FoV (deg)")
    args = ap.parse_args()
    cam = CameraModel(W, H, vfov_deg=args.vfov)
    print(f"Camera: {W}x{H}  HFoV {cam.hfov_deg:.1f}  VFoV {cam.vfov_deg:.1f}  "
          f"fpx_v {cam.fpx_v:.0f}  hold_range {cam.hold_range(0.30):.1f} m")
    track_crossing_envelope(args.vfov)
    dive_ground_envelope(args.vfov)
    dive_altitude_geometries(args.vfov)
    vfov_sensitivity()
    return 0


if __name__ == "__main__":
    sys.exit(main())
