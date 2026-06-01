# How pi-fpv-companion works

A Raspberry Pi Zero 2 W with an on-sensor-AI camera (IMX500) rides on an FPV quad,
**sees** a target, and **flies the aircraft toward it** — with **no GPS**, by
commanding body rates + thrust to the flight controller (in GUIDED_NOGPS). The
pilot stays in command: a 3-position switch (ch7) arms it, and flipping the FC-mode
channel out of GUIDED_NOGPS hands control straight back.

This doc explains the whole chain end to end. For the *why* of the design choices
see `architecture-audit.md` and `gps-denied-modes.md`; for flight gates see
`deployment-safety.md`.

---

## TL;DR

- The Pi is the camera **and** the brain. Its composite video (with a target box
  drawn on it) feeds the FC's camera input → the FC's OSD → VTX → goggles.
- Each frame: **detect → track → filter → compute a steering intent → safety-gate
  it → send it to the FC**.
- The FC link is **`SET_ATTITUDE_TARGET`** — the Pi commands body **rates** + real
  thrust while the FC sits in **GUIDED_NOGPS** (`control_mode: guided_nogps`, the
  deploy default). The rates are integrated by the airframe, so a noisy detector box
  yields smooth motion. No GPS. STABILIZE / ALT_HOLD + `RC_CHANNELS_OVERRIDE` remain
  as fallbacks.
- A 3-position switch (ch7) picks **STANDBY** / **TRACK** (follow + hold range) /
  **DIVE** (commit: pursuit guidance onto the target). A separate target-select
  input (ch9) cycles the lock among detections. In guided_nogps, STANDBY holds a
  level hover; manual recovery is the pilot flipping the FC-mode channel out of
  GUIDED_NOGPS.
- GUIDED_NOGPS requires `GUID_OPTIONS` bit 3 (ThrustAsThrust); the preflight param
  check sets+verifies it (without it the FC reads thrust as a climb-rate and the
  dive planes). On the STABILIZE fallback, the companion runs its own **adaptive
  hover** (a vertical-velocity loop that learns the hover throttle from
  `VFR_HUD.climb`).

---

## Hardware & video path

```
 IMX500 sensor (on-chip detection)
        │  frames + detections
        ▼
 Pi Zero 2 W  ──draw target box──►  CVBS composite out ──► FC camera-in
        │                                                      │
        │  MAVLink over UART (115200, pin8 TXD→FC RX,          │ FC overlays its
        │                     pin10 RXD→FC TX)                 │ own flight OSD
        ▼                                                      ▼
 Flight controller (ArduCopter)  ◄──SET_ATTITUDE_TARGET───     VTX ──► goggles
```

The Pi **replaces** the analog FPV camera — there is no second camera being
tapped. The FC adds its own OSD (battery/attitude) on top of the Pi's video.
Two wires to the FC: the **UART** (MAVLink, for control + telemetry) and the
**composite video** line. See `hardware.md`.

---

## The big picture (per-frame data flow)

```
camera ─► [detector] ─► tracker ─► alpha-beta filter ─► visual servo ─► safety gate ─► FC backend
 frame    detections   one target   smoothed + quality   GuidanceIntent   muted?        RC override
                                                          (roll,pitch,                    or release
                                                           yaw_rate,thrust)
```

Everything is a swappable Protocol (`Camera`, `Detector`, `Tracker`,
`FlightController`), so the **same `Pipeline`** runs in three places unchanged:
- **Mac dev:** SyntheticCamera + UDP-loopback fake FC + a cv2 viewer.
- **SITL:** SyntheticCamera + real ArduCopter (Docker) over TCP.
- **Aircraft:** IMX500 camera + real FC over UART + composite output.

Driven by `main.py` (a factory that builds each component from config) and
`pipeline.py` (`Pipeline.tick()` is one iteration of the loop above).

---

## The pipeline, stage by stage

### 1. Camera → `FrameBundle`
`camera.frames()` yields a `FrameBundle(image, width, height, timestamp,
detections)`. Some cameras (IMX500, SyntheticCamera) emit detections **inline**
(the IMX500 on its sensor NPU). Others (file, webcam) yield raw frames and a
separate detector runs.

### 2. Detector (optional)
If the camera didn't already produce detections and a detector is configured, it
runs inline. Light dev detectors: `ColorBlobDetector`, `HaarFaceDetector`,
ArUco. Flight uses the IMX500's on-sensor model.

### 3. Tracker → one `Target`
`tracker.consume(image, detections, t)` turns a stream of detections into a single
locked target with a stable `track_id`. Default is `IouAssociator` (IoU matching +
lost-frame reacquire) — for a moving FC + moving target, association beats a
single-object correlation tracker. `ClassicalCv2Tracker` (MOSSE/KCF/CSRT) is also
available.

### 4. Alpha-beta filter → `FilteredTarget` (+ quality)
`AlphaBetaTargetFilter` smooths position, estimates image-plane velocity (for the
servo's feedforward), and assigns a **quality** 0..1. Quality collapses on the
failure modes the raw tracker can't see: implausible centroid jumps
(misdetection), class flips (locked a person, now it's a chair), confidence decay.
Everything downstream uses the *filtered* target, never the raw tracker output.

### 5. Visual servo → `GuidanceIntent`
`compute_intent(target, ServoConfig, mode)` turns pixel geometry into a
backend-agnostic command:

```
GuidanceIntent(roll_deg, pitch_deg, yaw_rate_dps, thrust, timestamp)
```

- **Yaw (both modes):** horizontal pixel offset from center → yaw rate
  (P gain + velocity feedforward), with a deadzone and a clamp. Keeps the target
  centered. Roll stays ~0 (turns are flown with yaw, pure-pursuit style).
- **TRACK:** pitch regulates **range** to hold the **distance at engagement** — it
  captures the gap when you flick to TRACK and keeps it (it maintains, never closes
  in). The error is range-linear — `engage_setpoint − 1/size_frac` (apparent size
  is ∝ 1/range, so its inverse tracks range): drifted farther → nose down (chase),
  too close → ease off / nose up (a collision guard, not a ram gain). A **PI** loop
  — proportional plus an integral with back-calculation anti-windup — holds that
  distance *exactly* even on a target moving away (pure-P would settle farther
  back). `thrust = 0.5` (hold altitude); it follows and holds distance, never dives.
- **DIVE:** closed-loop constant-bearing homing. A forward lean closes the gap —
  **steep** (fast) diving onto a below target, **gentle** when level/climbing
  toward an above one (so it stays framed) — and a commanded vertical **rate**
  (`vertical_rate_mps`, tracked by the backend on `VFR_HUD.climb`) holds the
  target's vertical **frame position**. Holding a fixed frame point is a constant
  bearing → a collision course, so the flight path follows the line of sight and
  moves altitude **onto** the target — descend onto one below, hold for one level,
  climb toward one above. Gated on horizontal aim; never pitches nose-up. See
  `docs/dive-guidance.md` (incl. the fixed-camera FOV limits).

Sign conventions: `pitch_deg < 0` = nose-down = forward; `yaw_rate_dps > 0` = yaw
right; `thrust 0.5` = hold. `yaw_sign`/`pitch_sign` exist to correct a mirrored or
rotated camera mount.

### 6. Safety gate
`gate(intent, target, switch, armed, now, SafetyConfig)` **mutes** the command
(returns `ZERO_INTENT`) unless every condition holds: switch engaged, FC armed,
target fresh (watchdog), and track quality above `min_track_quality`. Returns a
`GateResult(intent, muted, reason)`. This is the wrong-target / stale-lock guard.

### 7. FC backend
The pipeline then does the key branch:

```python
# guided_nogps: STANDBY = hold a level hover; TRACK/DIVE = send body rates.
# fallbacks: STANDBY = release() (hand sticks back); else send_intent().
```

---

## The control model: body-rates into GUIDED_NOGPS

A bare FPV quad has no GPS/EKF position estimate, so ArduPilot's velocity/position
GUIDED modes won't even arm. **GUIDED_NOGPS** does not need a position estimate —
it accepts `SET_ATTITUDE_TARGET` (body rates + thrust). With `control_mode:
guided_nogps` (the deploy default) `ArduPilotBackend`:

- **`send_body_rates(roll,pitch,yaw,thrust)`** → sends `SET_ATTITUDE_TARGET` with
  body **rates** + a real **thrust** field. The airframe integrates the rates, so a
  jittery detector box becomes smooth motion. The control law (TRACK range-hold,
  DIVE pursuit guidance onto the line-of-sight) is in `guidance/rate_control.py`.
- **STANDBY** is handled by the caller: in GUIDED_NOGPS the companion commands a
  **level hover** (it never coasts on the last attitude). Manual recovery is the
  pilot flipping the FC-mode channel **out of** GUIDED_NOGPS, after which
  `control_ready()` is false and the companion commands nothing.

### `GUID_OPTIONS` bit 3 (ThrustAsThrust) — required
ArduCopter reads the `SET_ATTITUDE_TARGET` thrust field as a *climb-rate* unless
`GUID_OPTIONS` bit 3 (=8) is set, in which case "throttle 0" wrongly means "hold
altitude" and the dive planes. The preflight param check **sets+verifies** the bit
(`ensure_param_bits`, `GUID_OPTIONS_THRUST_AS_THRUST`; SITL readback=8).

### Fallbacks: RC override into STABILIZE / ALT_HOLD
Set `control_mode` to match the FC mode to fall back to stick injection:
- **`send_intent(intent)`** → `intent_to_rc_overrides()` maps the intent to AETR
  PWM and sends `RC_CHANNELS_OVERRIDE` on channels 1–4 (RCMAP roll/pitch/throttle/
  yaw), leaving the rest = 0 (released) so the pilot keeps the mode + engage
  switches.
- **`release()`** → sends 0 on all channels = "use the receiver" → instant manual
  handback.

AETR mapping & signs: roll/pitch → lean-**angle** stick (full deflection at
`angle_max_deg`, match the FC's `ANGLE_MAX`); yaw → **rate** stick (full deflection
at `pilot_yaw_rate_dps`); per-axis `*_sign` flips for TX/RCMAP/mount.
(`ArduCopterRcMapping` in `fc/ardupilot.py`.) The fallback `control_mode` picks how
thrust → throttle:
- **`stabilize`** — **direct** throttle. 0.5 → hover, full-down really cuts power →
  a **true steep dive** (~16 m/s, ~77° in SITL). No FC altitude hold.
- **`althold`** — throttle is a **climb rate** (0.5 = hold via baro), descent capped
  at `PILOT_SPEED_DN`. Gentle/altitude-safe, but can't dive hard.

### Adaptive hover (the STABILIZE fallback throttle "learns")
STABILIZE has no altitude hold, so on that fallback the companion provides one: a
**vertical-velocity PI loop** (`_adaptive_throttle`) reading `VFR_HUD.climb` from
the FC.
- **Kp** damps climb immediately (stops oscillation).
- **Ki** slowly trims a learned hover-throttle estimate — so you only give it a
  *rough* `stab_hover_throttle_us` seed and it converges to true hover by itself
  (SITL: seed 1300 → learned 1474, climb → 0 in ~6 s).
- It only learns while ~holding (`|thrust-0.5| < band`), is **frozen during a
  commanded dive**, is clamped to `[hover_min,max]`, and falls back to the fixed
  seed if `VFR_HUD` isn't arriving (it warns once). The telemetry streams
  (RC_CHANNELS + VFR_HUD) are re-requested every ~5 s so they survive link blips.

There is intentionally **no altitude floor** — a diving craft is supposed to lose
altitude; the pilot + engage switch are the backstop.

---

## Engagement: the channels (ch7 mode, ch9 select)

The Pi reads two RC channels back from the FC's `RC_CHANNELS`:

- **ch7** (`switch_channel`) — the 3-position **mode** switch, turned into a mode
  via two thresholds:

| switch     | mode     | what the companion does                                  |
|------------|----------|----------------------------------------------------------|
| low        | STANDBY  | in GUIDED_NOGPS, **hold a level hover** (fallbacks: `release()`) |
| mid        | TRACK    | follow: yaw to center the target, hold range            |
| high       | DIVE     | commit: pursuit guidance onto the target (descend/hold/climb) |

- **ch9** (`select_channel`) — **target-select**: a rising edge cycles the locked
  target among the detections (the `multi_iou` tracker).

The pilot's *separate* FC-mode channel selects GUIDED_NOGPS. (In SITL, `force_mode`
substitutes for the switch since there's no TX.)

Flip ch7 to TRACK and the craft starts chasing; flip to DIVE to commit; flip back
to STANDBY for a level hover. To take the sticks back, flip the FC-mode channel out
of GUIDED_NOGPS.

---

## Safety & failsafe model

1. **Muted unless engaged.** The gate mutes on disarm, stale target, or low track
   quality. In STANDBY (guided_nogps) the companion holds a level hover; on the
   fallbacks no commands flow.
2. **Manual recovery.** The pilot flips the FC-mode channel **out of** GUIDED_NOGPS;
   the companion then commands nothing (`control_ready()` is false) and the radio's
   sticks resume. (On the STABILIZE/ALT_HOLD fallbacks, STANDBY → `release()` hands
   the sticks straight back, since those are pilot modes.)
3. **Fail-safe on Pi death.** In guided_nogps the FC holds via ArduCopter's GUIDED
   command timeout; the companion also emits a ~1 Hz GCS heartbeat so `FS_GCS` is
   armed — for GPS-denied, set the GCS failsafe to **LAND** (RTL/SmartRTL need GPS).
   On the fallbacks, override frames stop and the FC's `RC_OVERRIDE_TIME` reverts the
   channels to the real radio within ~1–3 s.
4. **Camera watchdog.** If the camera stalls or never delivers a frame, the
   process exits for systemd to restart it.
5. **Closure limiting.** TRACK pitch is regulated by a range-linear error (from
   bbox size) toward the engage distance and reverses if the target gets closer
   than that — a collision guard, not a ram-the-target gain. The PI integral has
   back-calculation anti-windup and resets on a new lock / on leaving TRACK (along
   with the captured setpoint), so it can't carry stale lean across targets or modes.

The most dangerous misconfiguration is a **wrong stick sign** (divergent positive
feedback — the craft accelerates *away* from the target). It must be bench-verified
props-off; see `deployment-safety.md` §4.

---

## Why it works with no GPS

- GUIDED_NOGPS is an angle/rate submode that needs no position estimate; STABILIZE
  and ALT_HOLD need only **baro + IMU** — no GPS, no EKF origin, ever.
- `SET_ATTITUDE_TARGET` (rates + thrust) and `RC_CHANNELS_OVERRIDE` (stick
  injection) are both GPS-independent.
- The companion never asks the FC for position; it closes the loop **in image
  space** (pixels → rates / yaw / pitch).

SITL-proven with GPS fully disabled (`probe_nogps_modes.py`, 10/10). Details in
`gps-denied-modes.md`.

---

## Configuration (key knobs, `config/*.yaml`)

```yaml
fc:
  backend: ardupilot
  switch_channel: 7            # 3-position mode switch (STANDBY/TRACK/DIVE)
  select_channel: 9            # target-select (rising edge cycles the lock)
  track_threshold_us: 1300     # >= this -> TRACK
  dive_threshold_us: 1700      # >= this -> DIVE; else STANDBY
  control_mode: guided_nogps   # flight path: body-rates in GUIDED_NOGPS
                               #   (stabilize | althold = RC-override fallbacks)
  angle_max_deg: 45.0          # match the FC's ANGLE_MAX (fallback AETR mapping)
  pilot_yaw_rate_dps: 180.0    # full yaw stick rate
  rc_roll_sign: 1              # per-axis sign — BENCH VALIDATE
  rc_pitch_sign: 1
  rc_yaw_sign: 1
  stab_hover_throttle_us: 1450 # rough hover seed (adaptive hover refines it)
  stab_hover_learn: true       # vertical-velocity hold (needs VFR_HUD streamed)
  stab_hover_learn_kp: 50.0    # immediate climb damping
  stab_hover_learn_gain: 20.0  # Ki: slow hover trim
```

`config/imx500.yaml` is the flight airframe config and `config/mac-dev.yaml` the
laptop simulation.

---

## Validation status (honest)

- **Software:** 273 unit tests pass.
- **SITL + Gazebo (camera-in-the-loop), guided_nogps:** clean TRACK→DIVE→impact at
  25/40/55 m, a moving target, STANDBY safe-hold, Pi-death hold, and `GUID_OPTIONS`
  enforcement (readback=8).
- **SITL (ArduCopter 4.6.3, `docker/sitl-4.6/`), fallbacks:** control-sense 9/9,
  closed-loop chase PASS, dive comparison (STABILIZE ~16 m/s vs ALT_HOLD ~1–5),
  GPS-off modes 10/10, adaptive hover learns from a wrong seed. Scripts:
  `validate_sitl.py`, `fly_sitl.py`, `measure_dive_sitl.py`, `probe_nogps_modes.py`,
  `learn_hover_sitl.py`.
- **Hardware: not yet.** SITL is not a real airframe. Stick signs, hover-loop gains
  on real baro/propwash, VFR_HUD streaming, the detector against the real target,
  and the handback test must be validated **props-off then low/slow** before any
  committed flight. The pre-flight gates are in `deployment-safety.md`.

---

## File map

```
src/pi_fpv_companion/
  main.py            entry point; builds components from config
  pipeline.py        the per-frame loop (Pipeline.tick)
  types.py           GuidanceMode, GuidanceIntent, SwitchState, ...
  camera/            Camera Protocol + synthetic/file/webcam/imx500
  detect/            Detector Protocol + color/haar/aruco
  track/             Tracker Protocol + iou_associator/cv2_tracker + alpha-beta filter
  guidance/
    visual_servo.py  pixels -> GuidanceIntent (fallback AETR/attitude path)
    rate_control.py  guided_nogps body-rate control law (TRACK hold, DIVE pursuit)
    safety.py        the mute gate
  fc/
    base.py          FlightController Protocol
    ardupilot.py     backend: SET_ATTITUDE_TARGET rates + RC-override fallback  ← the core
    betaflight.py    MSP demo backend (RX-failsafe caveat — demo only)
  video/             overlay + framebuffer/DRM (TV out) sinks
docs/                this file + architecture-audit, gps-denied-modes, deployment-safety, sitl
scripts/             SITL validation + demos + Pi install
docker/sitl-4.6/     ArduCopter 4.6.3 SITL image
```
