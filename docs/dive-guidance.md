# DIVE / TRACK guidance, the fixed-camera FOV problem, and the agnostic dive

This explains how TRACK and DIVE steer the aircraft, the one constraint that
governs whether they keep their lock — **the camera is bolted to the airframe**
— and the altitude-agnostic dive built around it. It is grounded in a
closed-loop simulator (`tests/closed_loop_sim.py`) and SITL physics
(`scripts/measure_dive_sitl.py`, ArduCopter 4.6.3).

## The fixed-camera coupling

There is no gimbal. The camera looks straight out the nose, so **every yaw and
pitch command rotates the field of view**. That couples guidance and perception:

- Yaw to centre a target horizontally → the whole FOV swings; fine, the target
  moves toward centre.
- Lean forward (nose-down) to close distance → the boresight depresses, so a
  target ahead **rises in the frame**. Lean too hard and it leaves the top.
- Descend/climb → the line-of-sight (LOS) elevation to the target changes, so it
  drifts vertically in the frame even with no attitude change.

A single-frame test of `compute_intent` cannot see any of this — it needs the
loop closed (command → motion → new pixel position → command). That is what the
simulator does: a pinhole camera (IMX500 optics) → `AlphaBetaTargetFilter` →
`compute_intent` → `safety.gate` → airframe kinematics → repeat.

## Camera: Raspberry Pi AI Camera (Sony IMX500)

From the product brief: **HFoV 66.3°, VFoV 52.3°**, 4056×3040, f = 4.74 mm,
F1.79. The guidance frame is 720×576. The only place guidance converts pixels to
an angle is the dive's vertical LOS elevation, so it uses the **vertical** FoV
(`camera_vfov_deg`, default 52.3) — robust to whether the 4:3 sensor is scaled or
cropped into the frame, since the full height spans the VFoV either way.

**Acquisition cone.** A fixed forward camera can only *see* a target within
±VFoV/2 ≈ **±26°** of the boresight (plus whatever the nose is pitched). A ground
target is below, so from altitude `h` it is only visible once it is far enough
ahead that its depression drops under ~26°, i.e. horizontal range `> h / tan(26°)
≈ 2.05·h`. Steeper (nearer) ground targets are **not visible** to this camera —
no guidance change fixes that; it needs a tilted mount or a wider/downward lens.

## TRACK

Follow and hold range. Yaw is P + velocity feed-forward on the horizontal pixel
error; pitch is closure-regulated to `desired_bbox_frac` (bbox height is the
range proxy) plus a gentle vertical re-centre so the forward lean doesn't tip the
target out the top. Altitude is held (throttle neutral / adaptive hover).

TRACK keeps the target framed across the whole realistic crossing-speed envelope;
it only loses a target whose angular rate exceeds `max_yaw_rate_dps` (e.g. a fast
crosser at close range) — a physical limit, surfaced by the simulator, not a bug.

## DIVE — altitude-agnostic, LOS-elevation keyed

DIVE commits: it closes and, depending on where the target is, descends, holds,
or climbs. The target may be **below** (the usual ground case), roughly **level/
ahead**, or **above** — the logic must not assume.

### Why "keep it centred" fails on a ground target

Centring a target that is below you means pitching nose-down by the full LOS
depression. As you close, the depression grows; once it exceeds `max_pitch_deg`
the camera cannot look down far enough and the target falls out the **bottom**.
Worse, an aggressive throttle-cut descends far faster than the aircraft closes —
SITL shows STABILIZE drops ~16 m/s but only makes ~3–5 m/s forward at a 30° lean
— so the aircraft **pancakes into the ground short of the target**.

### The two ideas that fix it

1. **Bias the framing toward the dive's leading edge.** Instead of centring, hold
   the target offset toward the side the LOS is sweeping *away* from (high in the
   frame for a below target). This reserves frame for the depression to grow into
   *and* sustains the forward lean that closes. Set by `dive_vertical_bias_frac`.

2. **Key the direction on TRUE LOS elevation, not frame position.** The bias/
   descent direction comes from `aircraft_pitch_deg` (FC `ATTITUDE`) + the
   in-frame elevation. A ground target correctly framed *high* still reads
   "below the horizon", so the dive keeps diving instead of false-flipping to a
   climb. Absent fresh ATTITUDE, pitch falls back to 0 (level) — a safe
   degradation. This is what makes DIVE **altitude-aware** in all three regimes:
   - below the horizon → descend onto it (gravity dive) — the mission;
   - level → hold altitude, pursue horizontally (do **not** dive under it);
   - above → climb toward it (throttle) and keep it framed — but see the
     fixed-camera limitation below: aggressive closure on an above target is
     **not** supported.

3. **Geometry-match the descent.** The vertical commit scales with the LOS
   depression over `dive_los_band_deg`, so the flight path follows the LOS:
   gentle descent on a far/shallow target (no pancake), full commit on a steep/
   near one. A narrow band makes every dive an aggressive cut and pancakes.

`dive_pitch_up_max_deg` is **0** (shipped): DIVE is commit, so it never pitches
nose-up. Pitching up to frame an above target would point the velocity vector
backward and stall the closure; an above-target climb is the throttle's job.

### Two integration constraints that bit (and how they're handled)

- **The adaptive-hover hold band.** In STABILIZE the companion's adaptive-hover
  PI loop *holds altitude* whenever `|thrust-0.5| < hover_learn_band`. The
  geometry-matched dive only offsets thrust by `dive_descent` (~0.12), so the
  band **must** be below that (it is 0.05) or the hold loop silently cancels the
  descent and the aircraft never dives. The closed-loop sim models this band so
  the failure can't hide.
- **Above-target closure is geometrically limited.** A fixed forward camera can
  only keep an above target framed by pitching *up*, which drives the aircraft
  *backward* — the opposite of closing. So DIVE on an above target climbs toward
  it (throttle) and holds it framed, but cannot run it down. The committed strike
  is a *downward* attack: get above the target first. The closure also stalls a
  few degrees short of co-altitude where the gentle climb command falls back into
  the hold band — full robustness needs the deferred closed-loop flight-path
  control.

### Tuning (shipped, `config/imx500.yaml`)

| param | value | role |
|---|---|---|
| `dive_vertical_bias_frac` | 0.50 | bias setpoint toward the leading edge |
| `dive_los_band_deg` | 30.0 | depression band the descent/climb ramps over |
| `dive_descent` | 0.12 | throttle delta at full commit (0.5 ± this); **must exceed `hover_learn_band` = 0.05** |
| `dive_forward_deg` | 12.0 | forward (nose-down) lean while diving |
| `dive_pitch_up_max_deg` | 0.0 | never pitch nose-up (commit; nose-up = backward) |
| `camera_vfov_deg` | 52.3 | IMX500 vertical FoV; **must match the lens** |

Tuned in the closed-loop sim against a SITL-grounded airframe (`v_climb_max` 16,
`drag` 1.1). Bench/SITL-validate before flight; these are model values.

## Validated envelope

- **TRACK**: keeps the target framed for all reasonable crossing speeds; converges
  to the closure hold range (~6 m for a 1.7 m subject).
- **DIVE direction & framing**: correct for below / level / above — the target
  stays in frame and altitude moves the right way (descend / hold / climb), and it
  never dives away from or flies backward at an above target.
- **DIVE closure**: robust at **20–35 m engagement altitude** across the whole
  acquirable cone. Closure is forward-speed limited (~4 m/s on the grounded
  airframe), so far/shallow high-altitude dives close slowly; full robustness
  there needs **closed-loop flight-path-angle control** (descent regulated on
  `VFR_HUD.climb` against the LOS), tracked as `dynamic-vertical-control` and not
  yet implemented. There is **no automatic pull-up** — the pilot ends the dive
  with the flight-mode switch.

## Reproduce

```
# Envelope tables (no hardware): TRACK crossing-FOV, DIVE ground envelope,
# below/level/above outcomes, lens-VFoV sensitivity.
.venv/bin/python scripts/sim_track_dive.py
.venv/bin/python scripts/sim_track_dive.py --vfov 40   # narrower-lens stress

# Closed-loop property tests
.venv/bin/python -m pytest tests/test_closed_loop_sim.py tests/test_visual_servo.py

# SITL dive physics (ArduCopter 4.6.3 container)
.venv/bin/python scripts/measure_dive_sitl.py --connect tcp:127.0.0.1:5760
```

See also: `gps-denied-modes.md` (why STABILIZE for the dive),
`deployment-safety.md` (sign self-test), `architecture-audit.md` (pipeline).
