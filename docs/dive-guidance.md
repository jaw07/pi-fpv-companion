# DIVE / TRACK guidance, the fixed-camera FOV problem, and closed-loop homing

This explains how TRACK and DIVE steer the aircraft, the one constraint that
governs whether they keep their lock — **the camera is bolted to the airframe**
— and the closed-loop constant-bearing dive built around it. It is grounded in a
closed-loop simulator (`tests/closed_loop_sim.py`) and SITL physics
(`scripts/measure_dive_sitl.py`, `scripts/validate_vrate_sitl.py`, ArduCopter
4.6.3).

## The fixed-camera coupling

There is no gimbal. The camera looks straight out the nose, so **every yaw and
pitch command rotates the field of view**. That couples guidance and perception:

- Yaw to centre a target horizontally → the whole FOV swings; the target moves
  toward centre.
- Lean forward (nose-down) to close → the boresight depresses, so a target ahead
  **rises in the frame**. Lean too hard and it leaves the top.
- Descend/climb → the line-of-sight (LOS) elevation to the target changes, so it
  drifts vertically in the frame even with no attitude change. (Descending raises
  a below target in frame; climbing lowers it.)

A single-frame test of `compute_intent` cannot see any of this — it needs the
loop closed (command → motion → new pixel position → command). That is what the
simulator does: a pinhole camera (IMX500 optics) → `AlphaBetaTargetFilter` →
`compute_intent` → `safety.gate` → airframe kinematics → repeat.

## Camera: Raspberry Pi AI Camera (Sony IMX500)

From the product brief: **HFoV 66.3°, VFoV 52.3°**, 4056×3040, f = 4.74 mm,
F1.79. The guidance frame is 720×576.

**Acquisition cone.** A fixed forward camera can only *see* a target within
±VFoV/2 ≈ **±26°** of the boresight (plus whatever the nose is pitched). A ground
target is below, so from altitude `h` it is only visible once it is far enough
ahead that its depression drops under ~26°, i.e. horizontal range `> h / tan(26°)
≈ 2.05·h`. Steeper (nearer) ground targets are **not visible** to this camera —
no guidance change fixes that; it needs a tilted mount or a wider/downward lens.

## TRACK

Follow and hold range. Yaw is P + velocity feed-forward on the horizontal pixel
error; pitch is a **PI** closure loop that holds a fixed standoff, plus a gentle
vertical re-centre so the forward lean doesn't tip the target out the top.
Altitude is held (throttle neutral / adaptive hover) — TRACK follows and keeps
its distance, it never dives.

The closure error is **range-linear**, not raw apparent size. Apparent size is
∝ 1/range, so the controller regulates `1/desired_bbox_frac − 1/size_frac`
(≈ `hold_range − range`): this conditions the loop identically at every distance,
where a raw-size error would make a far target sluggish and an integral on top of
it slow-oscillate. The **integral** (with back-calculation anti-windup, reset per
lock / on leaving TRACK) drives the steady-state range to *exactly*
`desired_bbox_frac` on a target moving away — pure-P alone settles farther back,
because a residual size error is needed to sustain the chase lean. Sim: a target
receding at 1 m/s is held at +0.1 m of the hold distance with no limit cycle,
versus several metres of lag for pure-P. `closure_i_gain = 0` selects pure-P.

TRACK keeps the target framed across the whole realistic crossing-speed envelope;
it only loses a target whose angular rate exceeds `max_yaw_rate_dps` (e.g. a fast
crosser at close range) — a physical limit, surfaced by the simulator, not a bug.

## DIVE — closed-loop constant-bearing homing

DIVE commits and moves altitude onto the target. The target may be **below** (the
usual ground case), roughly **level/ahead**, or **above** — the logic must not
assume.

### Why a fixed forward camera makes this hard

The camera couples **pitch** (which drives forward closure *and* aims the camera
vertically) with the need to keep the target framed. Lean forward to close → the
boresight depresses → the target rises in frame. To keep a target centred with
pitch you'd have to pitch *up* for an above target, which points the velocity
vector backward and stalls the approach. Pitch alone cannot both close and frame.

### The decoupling: pitch closes, throttle frames

DIVE breaks the coupling:

- **PITCH** is a forward (nose-down) commit lean, **adaptive** to the engagement:
  *steep* (`dive_forward_deg` ≈ 25°) when descending onto a target **below** the
  flight path — a fast, committed ground attack, and a steep nose-down also aims
  the fixed camera down at the target, keeping it framed — but *gentle*
  (`dive_climb_forward_deg` ≈ 6°) when level/climbing toward an **above** target,
  where a steep lean would push it out the top faster than the (gravity-limited)
  climb can re-centre it. It ramps gentle→steep with the commanded descent, is
  clamped nose-down by DIVE's own steeper `dive_max_pitch_deg` (never backs off /
  pitches up). This makes a ground attack fast (~20–30 s to impact in sim, ~3× the
  gentle lean) without losing an above target. SITL-confirmed: STABILIZE tracks the
  steep RC-override lean within ~1° (cmd −25° → −24°, −30° → −29°). A **soft-start**
  (`dive_lean_ramp_s`) ramps the steep lean in over ~0.5 s at commit so the target
  doesn't slew across the frame faster than the tracker/filter can follow (without
  it, a snap to full lean briefly out-runs the velocity estimate → a momentary
  tracking hiccup at commit).
- **THROTTLE** flies a commanded vertical **rate** that holds the target's
  vertical **frame position**. The servo emits `GuidanceIntent.vertical_rate_mps`
  (+up); the ArduPilot backend's climb-rate PI loop tracks it against
  `VFR_HUD.climb`. Below centre (target drifted low) → descend (raises it back);
  above centre → climb (lowers it).

Holding a target at a fixed point in the frame is a **constant bearing**, which
(constant-bearing-decreasing-range) is a **collision course**. So the flight path
automatically follows the line of sight — **descend onto a below target, hold for
a level one, climb toward an above one** — with no attitude or FoV input needed.
The frame error *is* the signal.

Two nested loops: the servo's framing loop (outer) commands the rate; the backend
(inner) tracks it. The inner P-loop has steady-state droop (SITL: −3 m/s command →
~−2.2 m/s); the outer loop integrates that out by commanding more.

The vertical commit is gated on **horizontal aim** (`dive_center_frac`): centre
yaw before committing power, so it doesn't dive off to the side.

### Tuning (shipped, `config/imx500.yaml`)

| param | value | role |
|---|---|---|
| `dive_forward_deg` | 25.0 | STEEP lean at full descent (fast ground attack) |
| `dive_climb_forward_deg` | 6.0 | gentle lean when level/climbing (keeps an above target framed) |
| `dive_max_pitch_deg` | 30.0 | DIVE nose-down clamp (steeper than TRACK's `max_pitch_deg`) |
| `dive_lean_ramp_s` | 0.5 | soft-start: ramp the steep lean in over this many s at commit |
| `dive_vrate_gain` | 17.0 | m/s of climb command per unit normalised vertical frame error |
| `dive_max_descent_mps` | 8.0 | clamp on commanded descent |
| `dive_max_climb_mps` | 4.0 | clamp on commanded climb (gravity-limited, < descent) |
| `dive_center_frac` | 0.30 | horizontal aim tolerance before committing vertical |

`dive_vrate_gain = 0` disables vertical homing (DIVE just leans in). Tuned in the
closed-loop sim against a SITL-grounded airframe; bench/SITL-validate before
flight.

### Operational dependency

The closed-loop dive needs **`VFR_HUD.climb` streaming** (`SR*_EXTRA2`) so the
backend can close the rate loop, and the loop must be drained every tick (the
pipeline does this via `read_switch`). Without fresh climb telemetry the backend
falls back to an open-loop throttle map (degraded but still descends). There is
**no automatic pull-up** — the pilot ends the dive with the flight-mode switch.

## Validated envelope

- **TRACK**: keeps the target framed for all reasonable crossing speeds; the PI
  closure converges to the hold range (`desired_bbox_frac` 0.15 ≈ 11.5 m for a
  1.7 m subject) and holds it on a receding target to within ~0.1 m (no limit
  cycle), versus several metres of lag for pure-P.
- **DIVE**: closes onto a target **below, level, or above** — sim reaches impact
  for far ground (140 m), level, and a +25 m above target at 100 m; with vertical
  homing OFF the same dives pancake or lose the target. Bounded by the fixed-camera
  acquisition cone (a steeply-below ground target isn't visible) and, for ground
  targets, by engagement altitude (robust at moderate altitude; very far/shallow
  high dives are forward-speed limited).
- A terminal frame-exit *inside the impact radius* is the target passing the
  camera at impact, not a tracking loss.
- **Perception robustness** (sim, the defences that matter for the dominant
  "confidently-wrong track" hazard): the loop rides out detection noise (≥12 px),
  dropout (≥30%), and detector latency (~5 frames) by smoothing/coasting; a
  **misdetection** (centroid teleport) is innovation-gated → quality collapses →
  the safety gate **mutes** (the aircraft holds, does not chase it) and recovers;
  a **class flip** is class-consistency-gated → mutes. An **occlusion** (target
  behind cover for seconds) → the filter coasts, quality decays, the gate mutes
  (the aircraft holds, doesn't fly blind), and on reappearance it **re-acquires**
  and the dive resumes. A seeded Monte-Carlo over randomized noisy ground
  engagements hits ~93% (miss-distance p90 ~1.5 m).
- **Tracker association** is IoU **or centroid-distance** gated, matched against a
  constant-velocity **prediction**: a distant target is a tiny box (a person at
  >100 m is a few px wide), so under camera rotation it shifts more than its own
  width → zero IoU; distance gating keeps the lock pure IoU would drop every frame.
  Matching the prediction (not the last position) also keeps identities through a
  **crossing** — two targets passing in the image would otherwise swap ids (and
  the lock would follow the wrong one). `iou` (single) and `multi_iou` (multi).
  Limit: a crossing *with heavy detection noise + dropout* is genuinely ambiguous
  — id-preservation is ~100% clean, ~75% under heavy degradation (a fundamental
  data-association limit). Pick your target when candidates are well separated.
- **Crossing speed**: a ground attack is fast (steep lean), but a target
  *translating laterally* faster than the aircraft's forward speed stays framed
  (yaw keeps up) yet isn't run down — a kinematic limit (you can't catch what's
  faster than you), not a guidance bug. The mission target (ground, static/slow)
  is well within it; `lead_time_s` helps the intercept geometry.

## Reproduce

```
# Envelope tables (no hardware): TRACK crossing-FOV, DIVE homing OFF vs ON,
# below/level/above outcomes, lens-VFoV sensitivity.
.venv/bin/python scripts/sim_track_dive.py
.venv/bin/python scripts/sim_track_dive.py --vfov 40   # narrower-lens stress

# Closed-loop property tests
.venv/bin/python -m pytest tests/test_closed_loop_sim.py tests/test_visual_servo.py

# SITL (ArduCopter 4.6.3 container)
.venv/bin/python scripts/measure_dive_sitl.py       --connect tcp:127.0.0.1:5760  # dive physics
.venv/bin/python scripts/validate_vrate_sitl.py     --connect tcp:127.0.0.1:5760  # rate-loop tracking
.venv/bin/python scripts/validate_steep_dive_sitl.py --connect tcp:127.0.0.1:5760  # steep-lean tracking
```

See also: `gps-denied-modes.md` (why STABILIZE for the dive),
`deployment-safety.md` (sign self-test), `architecture-audit.md` (pipeline).
