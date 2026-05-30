# How pi-fpv-companion works

A Raspberry Pi Zero 2 W with an on-sensor-AI camera (IMX500) rides on an FPV quad,
**sees** a target, and **flies the aircraft toward it** — with **no GPS**, by
impersonating the pilot's sticks to the flight controller. The pilot stays in
command: a 3-position switch arms it, and letting go hands control straight back.

This doc explains the whole chain end to end. For the *why* of the design choices
see `architecture-audit.md` and `gps-denied-modes.md`; for flight gates see
`deployment-safety.md`.

---

## TL;DR

- The Pi is the camera **and** the brain. Its composite video (with a target box
  drawn on it) feeds the FC's camera input → the FC's OSD → VTX → goggles.
- Each frame: **detect → track → filter → compute a steering intent → safety-gate
  it → send it to the FC**.
- The FC link is **`RC_CHANNELS_OVERRIDE`** — the Pi writes the roll/pitch/yaw/
  throttle (AETR) stick values, while the FC sits in a normal **STABILIZE** (or
  ALT_HOLD) pilot mode. No GPS, no GUIDED, no special firmware mode.
- A 3-position switch picks **STANDBY** (Pi muted, pilot flies) / **TRACK**
  (follow + hold range/altitude) / **DIVE** (commit: close and move altitude onto
  the target — descend / hold / climb by where it is).
- STABILIZE has no altitude hold, so the companion runs its own **adaptive hover**
  (a vertical-velocity loop that learns the hover throttle from `VFR_HUD.climb`).

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
 Flight controller (ArduCopter)  ◄──RC_CHANNELS_OVERRIDE──     VTX ──► goggles
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
(on-sensor NPU). Others (PiCam, file, webcam) yield raw frames and a separate
detector runs.

### 2. Detector (optional)
If the camera didn't already produce detections and a detector is configured, it
runs — **async by default** (`AsyncDetector` on a pinned worker thread) so a slow
inference call doesn't stall the 30 FPS loop. Detectors: `ColorBlobDetector`,
`HaarFaceDetector`, `NanoDetDetector` (NCNN), `Yolov8Detector`. Flight uses the
IMX500's on-sensor model.

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
- **TRACK:** pitch regulates **range** — bbox size vs `desired_bbox_frac`: too far
  → nose down (accelerate forward), too close → ease off / nose up (a collision
  guard, not a ram gain). `thrust = 0.5` (hold altitude).
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
if switch.mode is STANDBY:   self._fc.release()        # hand sticks to the pilot
else:                        self._fc.send_intent(gated.intent)
```

---

## The control model: RC override into STABILIZE

A bare FPV quad has no GPS/EKF position estimate, so ArduPilot's velocity/position
GUIDED modes won't even arm. The companion instead writes **stick values** the FC
already understands, in a normal self-levelling **pilot** mode. `ArduPilotBackend`:

- **`send_intent(intent)`** → `intent_to_rc_overrides()` maps the intent to AETR
  PWM and sends `RC_CHANNELS_OVERRIDE` on channels 1–4 (RCMAP roll/pitch/throttle/
  yaw), leaving 5–8 = 0 (released) so the pilot keeps the mode + engage switches.
- **`release()`** → sends 0 on all channels = "use the receiver" → instant manual
  handback.

### AETR mapping & signs
roll/pitch → lean-**angle** stick (full deflection at `angle_max_deg`, match the
FC's `ANGLE_MAX`). yaw → **rate** stick (full deflection at `pilot_yaw_rate_dps`).
Per-axis `*_sign` flips for TX/RCMAP/mount. (`ArduCopterRcMapping` in
`fc/ardupilot.py`.)

### `control_mode`: how thrust → throttle
Must match the FC's actual flight mode:
- **`stabilize` (default)** — **direct** throttle. 0.5 → hover, full-down really
  cuts power → a **true steep dive** (~16 m/s, ~77° in SITL). No FC altitude hold.
- **`althold`** — throttle is a **climb rate** (0.5 = hold via baro), descent
  capped at `PILOT_SPEED_DN`. Gentle/altitude-safe, but can't dive hard.

We default to STABILIZE because the mission is to **dive** onto a target;
ALT_HOLD's altitude controller fights that.

### Adaptive hover (the throttle "learns")
STABILIZE has no altitude hold, so the companion provides one: a **vertical-
velocity PI loop** (`_adaptive_throttle`) reading `VFR_HUD.climb` from the FC.
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

## Engagement: the 3-position switch

The Pi reads one RC channel (`switch_channel`, default ch7) back from the FC's
`RC_CHANNELS`, and turns its PWM into a mode via two thresholds:

| switch     | mode     | what the companion does                                  |
|------------|----------|----------------------------------------------------------|
| low        | STANDBY  | `release()` — pilot has full manual control              |
| mid        | TRACK    | follow: yaw to center the target, hold range + altitude  |
| high       | DIVE     | commit: close + move altitude onto the target (descend/hold/climb) |

(In SITL, `force_mode` substitutes for the switch since there's no TX.)

The pilot flies normally in STABILIZE. Flip to TRACK and the craft starts chasing;
flip to DIVE to commit; flip back to STANDBY (or change the FC flight-mode switch)
and you instantly have the sticks again.

---

## Safety & failsafe model

1. **Muted unless engaged.** No commands flow in STANDBY; the gate also mutes on
   disarm, stale target, or low track quality.
2. **Instant handback.** STANDBY → `release()` → the radio's sticks resume. It's a
   pilot mode, not GUIDED, so there's no "ignore pilot / hover on timeout" lockout.
3. **Fail-safe on Pi death.** If the Pi process dies mid-engagement, override
   frames stop and the FC's `RC_OVERRIDE_TIME` reverts the channels to the real
   radio within ~1–3 s.
4. **Camera watchdog.** If the camera stalls or never delivers a frame, the
   process exits for systemd to restart it.
5. **Closure limiting.** TRACK pitch is regulated by bbox size and reverses past
   the target fraction — a collision guard, not a ram-the-target gain.

The most dangerous misconfiguration is a **wrong stick sign** (divergent positive
feedback — the craft accelerates *away* from the target). It must be bench-verified
props-off; see `deployment-safety.md` §4.

---

## Why it works with no GPS

- STABILIZE and ALT_HOLD need only **baro + IMU** — no GPS, no EKF origin, ever.
- `RC_CHANNELS_OVERRIDE` is plain stick injection — also GPS-independent.
- The companion never asks the FC for position; it closes the loop **in image
  space** (pixels → yaw/pitch) and altitude via baro climb rate.

SITL-proven with GPS fully disabled (`probe_nogps_modes.py`, 10/10). Details in
`gps-denied-modes.md`.

---

## Configuration (key knobs, `config/*.yaml`)

```yaml
fc:
  backend: ardupilot
  switch_channel: 7            # 3-position engage switch
  track_threshold_us: 1300     # >= this -> TRACK
  dive_threshold_us: 1700      # >= this -> DIVE; else STANDBY
  control_mode: stabilize      # stabilize (dive) | althold (altitude-safe)
  angle_max_deg: 45.0          # match the FC's ANGLE_MAX
  pilot_yaw_rate_dps: 180.0    # full yaw stick rate
  rc_roll_sign: 1              # per-axis sign — BENCH VALIDATE
  rc_pitch_sign: 1
  rc_yaw_sign: 1
  stab_hover_throttle_us: 1450 # rough hover seed (adaptive hover refines it)
  stab_hover_learn: true       # vertical-velocity hold (needs VFR_HUD streamed)
  stab_hover_learn_kp: 50.0    # immediate climb damping
  stab_hover_learn_gain: 20.0  # Ki: slow hover trim
```

`config/default.yaml` is the base, `config/imx500.yaml` the flight airframe,
`config/mac-dev.yaml` the laptop simulation.

---

## Validation status (honest)

- **Software:** 177 unit tests pass.
- **SITL (ArduCopter 4.6.3, `docker/sitl-4.6/`):** control-sense 9/9, closed-loop
  chase PASS, dive comparison (STABILIZE ~16 m/s vs ALT_HOLD ~1–5), GPS-off modes
  10/10, adaptive hover learns from a wrong seed. Scripts: `validate_sitl.py`,
  `fly_sitl.py`, `measure_dive_sitl.py`, `probe_nogps_modes.py`, `learn_hover_sitl.py`.
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
  camera/            Camera Protocol + synthetic/file/webcam/picam/imx500
  detect/            Detector Protocol + color/haar/nanodet/yolov8 + async wrapper
  track/             Tracker Protocol + iou_associator/cv2_tracker + alpha-beta filter
  guidance/
    visual_servo.py  pixels -> GuidanceIntent (TRACK vs DIVE)
    safety.py        the mute gate
  fc/
    base.py          FlightController Protocol
    ardupilot.py     RC-override backend, AETR mapping, adaptive hover  ← the core
    betaflight.py    MSP demo backend (RX-failsafe caveat — demo only)
  video/             overlay + framebuffer/DRM (TV out) sinks
docs/                this file + architecture-audit, gps-denied-modes, deployment-safety, sitl
scripts/             SITL validation + demos + Pi install
docker/sitl-4.6/     ArduCopter 4.6.3 SITL image
```
