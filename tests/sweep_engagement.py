"""Engagement matrix sweep (diagnostic, not a pytest).

Runs the full TRACK->DIVE chain through the closed-loop sim across varying
altitudes, target physical sizes, and target motion, and prints a table of the
behaviours the user cares about:

  * TRACK keeps the target in frame (no loss before the dive)
  * TRACK pitch is steady (no nod): nod = max-min pitch over the TRACK phase
  * DIVE reaches the target (min_range) and hits centre mass (terminal px offset)

Run:  .venv/bin/python -m tests.sweep_engagement
"""
from __future__ import annotations
import math

from pi_fpv_companion.types import GuidanceMode
from tests.closed_loop_sim import Airframe, CameraModel, SimWorld, imx500_servo, imx500_safety

W, H = 720, 576


def _run(alt, horiz, lateral, tgt_alt, th_m, tw_m, vel, track_s=4.0, dur=60.0, **servo_kw):
    cam = CameraModel(W, H, target_h_m=th_m, target_w_m=tw_m)
    world = SimWorld(
        camera=cam, servo=imx500_servo(**servo_kw), safety=imx500_safety(),
        airframe=Airframe(pos=(0.0, 0.0, alt)),
        target_pos=(horiz, lateral, tgt_alt), target_vel=vel,
    )
    tr = world.run(GuidanceMode.DIVE, duration_s=dur, dive_after_s=track_s)
    track = [tk for tk in tr.ticks if tk.t < track_s]
    track_inframe = [tk for tk in track if tk.in_frame]
    pitches = [tk.pitch_cmd for tk in track_inframe]
    nod = (max(pitches) - min(pitches)) if pitches else 0.0
    # reversals in TRACK pitch (sign changes of the slope)
    rev = 0
    for a, b, c in zip(pitches, pitches[1:], pitches[2:]):
        if (b - a) * (c - b) < 0:
            rev += 1
    # AIM centring: px offset at the last in-frame tick still ~5 m out (before the
    # target fills the frame, where pixels exaggerate a sub-metre miss). Falls back to
    # the last in-frame tick if it never reaches 5 m.
    inframe = [tk for tk in tr.ticks if tk.in_frame]
    aim_ticks = [tk for tk in inframe if tk.range_m > 5.0]
    ref = aim_ticks[-1] if aim_ticks else (inframe[-1] if inframe else None)
    term_off = math.hypot(ref.px - W / 2, ref.py - H / 2) if ref else float("nan")
    track_lost = any((not tk.in_frame) for tk in track) and track[0].in_frame
    depression = math.degrees(math.atan2(alt - tgt_alt, horiz))
    return dict(nod=nod, rev=rev, min_range=tr.min_range, term_off=term_off,
                track_lost=track_lost, ever_left=tr.lost_before_impact(4.0),
                depr=depression, p0=pitches[0] if pitches else float("nan"),
                pmax=max(pitches) if pitches else float("nan"),
                pmin=min(pitches) if pitches else float("nan"))


def main():
    print(f"{'scenario':42s} {'depr':>5s} {'p0':>6s} {'nod':>6s} {'rev':>4s} "
          f"{'minR':>6s} {'term':>6s} {'lost?':>6s}")
    print("-" * 92)
    # ground targets at varying altitude (horiz set to ~2.3*alt so depression ~23 deg,
    # within the downward FOV at level flight), person-sized
    for alt in (15, 25, 40, 55, 70):
        horiz = round(alt * 2.3)
        r = _run(alt, horiz, 0.0, 0.0, 1.7, 0.5, (0, 0, 0))
        _row(f"ground person  alt={alt:>2d} horiz={horiz:>3d}", r)
    print()
    # varying target SIZE at a fixed high engagement (40 m)
    for name, th, tw in (("small sign 0.4", 0.4, 0.4), ("person 1.7", 1.7, 0.5),
                         ("vehicle 1.6x4", 1.6, 4.0), ("big box 3x3", 3.0, 3.0)):
        r = _run(40, 92, 0.0, 0.0, th, tw, (0, 0, 0))
        _row(f"size {name:18s} alt=40 horiz=92", r)
    print()
    # MOVING ground targets (high engagement)
    for name, vel in (("static", (0, 0, 0)), ("recede +x 6", (6, 0, 0)),
                      ("approach -x 6", (-6, 0, 0)), ("cross +y 6", (0, 6, 0)),
                      ("cross +y 12", (0, 12, 0)), ("diag +x+y 5", (5, 5, 0))):
        r = _run(40, 92, 0.0, 0.0, 1.7, 0.5, vel)
        _row(f"move {name:18s} alt=40 horiz=92", r)
    print()
    # off-axis (lateral) ground targets at altitude
    for lat in (-25, -12, 12, 25):
        r = _run(40, 92, lat, 0.0, 1.7, 0.5, (0, 0, 0))
        _row(f"offaxis lat={lat:>4d}      alt=40 horiz=92", r)
    print()
    # SAME-altitude air targets (the gate must NOT regress these: full closure)
    for name, pos in (("air ahead 30", (30, 0, 50)), ("air ahead 55", (55, 0, 50)),
                      ("air above 15", (55, 0, 65)), ("air below 15", (55, 0, 35))):
        r = _run(50, pos[0], pos[1], pos[2], 1.7, 0.5, (0, 0, 0))
        _row(f"{name:18s}      alt=50", r)


def _row(label, r):
    flag = "LOST" if r["ever_left"] else ("track" if r["track_lost"] else "ok")
    print(f"{label:42s} {r['depr']:5.1f} {r['p0']:6.1f} {r['nod']:6.1f} {r['rev']:4d} "
          f"{r['min_range']:6.1f} {r['term_off']:6.1f} {flag:>6s}")


if __name__ == "__main__":
    main()
