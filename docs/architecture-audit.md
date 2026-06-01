# Architecture Audit (deep pass)

Three independent research reviews + a safety-model analysis. These are
*architectural* findings — they are not bugs in the code; the code largely
implements the current design correctly. The design itself has problems.

Severity legend: **FOUNDATIONAL** (design doesn't work on target hardware) /
**DEAD-END** (works but is the wrong architecture, don't invest further) /
**STRUCTURAL** (control-law deficiency, fixable) / **SAFETY** / **LATENT**.

---

> **STATUS: ADDRESSED in code + VALIDATED against a real ArduPilot flight
> stack.** `GuidanceIntent` is now attitude-domain (roll/pitch/yaw-rate/
> thrust); `visual_servo.py` emits it; `ardupilot.py` sends
> `SET_ATTITUDE_TARGET` (mask 3) for GUIDED_NOGPS; Betaflight remapped to
> ANGLE-mode sticks. 160/160 tests green.
>
> **SITL validation (2026-05-16): 10/10.** `scripts/validate_sitl.py` drove
> the *real* `ArduPilotBackend.send_intent()` path against ArduCopter **4.0.3**
> SITL (`radarku/ardupilot-sitl`, amd64 under emulation). Confirmed: the build
> accepts `GUIDED_NOGPS`; arms in it; the `SET_ATTITUDE_TARGET` (mask 3) stream
> is accepted with no rejection / no mode kick / no disarm across the whole
> run; and the copter tracks every axis in the correct sense — cmd +30 dps yaw
> → +29.1 measured, −30 → −28.4; cmd pitch −10° (nose-down/approach) → −9.5°;
> cmd roll +12° → +11.5° (quaternion sign correct). The velocity→attitude
> pivot is no longer just research + unit tests + a loopback echo.
>
> SITL also caught a real **doc/demo factual error** (now fixed in
> `docs/sitl.md` + the SITL scripts): `AHRS_EKF_TYPE 3` is *EKF3*, not
> "DCM-only"; on a build with EK3 disabled it hard-fails arming
> (`PreArm: no EKF3 cores`). Point AHRS at whichever EKF actually has cores
> (4.0.3 → EKF2). The control surface is independent of EKF flavour/GPS.
>
> Still hardware-gated (cannot be done in SITL — see `docs/deployment-safety.md`):
> GUIDED_NOGPS arming on a real GPS-denied airframe, the switch-as-flight-mode-
> channel wiring, thrust/`GUID_OPTIONS` altitude semantics, and re-confirming
> on the exact flight firmware (4.0.3 is older than likely flight builds; EKF3
> defaults and minor `SET_ATTITUDE_TARGET` handling differ on 4.5/4.6).

## 1. FOUNDATIONAL — ArduPilot velocity-in-GUIDED does not work on a GPS-denied FPV quad

> **UPDATE (2026-05-23):** the control path moved one step further — from
> GUIDED_NOGPS + `SET_ATTITUDE_TARGET` to **ALT_HOLD + `RC_CHANNELS_OVERRIDE`**
> (AETR sticks). Both are GPS-free (a GPS-disabled SITL arms+steers in either —
> `scripts/probe_nogps_modes.py`), but ALT_HOLD is a *pilot* mode: releasing the
> override hands control straight back to the pilot (a Pi crash fail-safes via the
> FC's RC-override timeout), avoiding GUIDED's "ignore pilot + hover on timeout"
> lockout. The velocity-vs-attitude analysis below still holds; we now command
> attitude as *sticks*, not `SET_ATTITUDE_TARGET`. See `docs/gps-denied-modes.md`.

The entire ArduPilot backend sends `SET_POSITION_TARGET_LOCAL_NED` velocity +
yaw-rate (`type_mask 0x5C7`). ArduPilot Copter `GUIDED` runs the position/
velocity controller in the EKF local frame and **requires an EKF position/
velocity estimate** (GPS, optical flow, VIO, or MoCap). A typical analog FPV
quad has none of these. Consequences:

- GUIDED won't even arm: `PreArm: Need Position Estimate`.
- A body-frame "velocity" command is *not* open-loop — ArduPilot closes the
  loop on measured velocity error. No estimate → no loop → command does nothing.
- The GPS-denied interface is **`GUIDED_NOGPS`, which accepts ONLY
  `SET_ATTITUDE_TARGET`** (roll/pitch angle + yaw-rate + thrust). It will not
  accept `SET_POSITION_TARGET_LOCAL_NED` or velocity in any form.

**Implication:** the control surface is wrong end-to-end. `visual_servo.py`
must emit attitude/rate intent (lean angle + yaw-rate + thrust), and
`ardupilot.py` must send `SET_ATTITUDE_TARGET` in `GUIDED_NOGPS`, not
velocity in `GUIDED`. (Velocity-in-GUIDED is only correct if a non-GPS
position source feeds `VISION_POSITION_ESTIMATE` — not in scope for a bare
FPV quad.) Firmware-version-sensitive: confirm body yaw-rate + thrust handling
in `SET_ATTITUDE_TARGET` on the exact Copter 4.x build.

Pilot-authority corollary: in GUIDED/GUIDED_NOGPS the pilot's roll/pitch/
throttle are **ignored** (only yaw maybe, per `GUID_OPTIONS`). The 3 s command
timeout makes it *hover*, it does **not** hand back to the pilot. Therefore
the arming switch MUST be the actual flight-mode channel, pilot-owned — so
releasing it instantly returns the aircraft to a pilot mode (AltHold). A
Pi-side software gate is insufficient; a Pi crash mid-engagement = blind
hover, pilot locked out, until the pilot flips the mode switch.

Betaflight `MSP_SET_RAW_RC` is worse: BF failsafe keys off the *RX* link, not
MSP. A hung Pi at full override with a healthy RX = frozen sticks, no
failsafe. Treat the Betaflight path as demo-only; never override arm/throttle.

## 2. CORRECTED — the Pi *is* the camera; CVBS to FC-cam-pad is the design

**Original framing was wrong** (operator correction). There is no separate
pilot camera being tapped/re-encoded. The Pi Zero is the *only* video source
— it replaces the analog FPV camera:

  Pi sensor -> detect -> draw target box -> CVBS out -> FC camera-in ->
  FC overlays its own flight OSD (battery/attitude) -> FC -> VTX -> goggles

This is a standard FC video pipeline with a smart camera substituted for a
dumb one. The MSP-DisplayPort recommendation does NOT apply — there is no
clean feed to overlay marks onto; the Pi must produce the actual pixels. The
`video/` subsystem (DRM/CVBS/composite) is the CORRECT flight architecture,
not a dead-end.

What remains true (downgraded DEAD-END -> inherent tradeoff):

- A dumb analog camera adds ~negligible latency; a Pi-as-camera adds sensor
  capture + libcamera + processing + CVBS encode (tens to >100 ms). This is the inherent cost of a smart
  camera doing onboard CV, not an architecture error. It degrades the
  *pilot's manual-flight feel* (hard to hand-fly for racing; fine for
  cruise/observe). It does NOT degrade the auto-engagement loop — during
  engagement the Pi closes the loop on its own fresh sensor frames, not the
  goggle feed.
- This is exactly why the IMX500 is the camera (30 FPS, on-sensor detect, much
  lower added latency); CPU-side detection was inadequate and was removed (§3) —
  but the *video architecture* is sound.
- Clean division of labor to preserve: Pi draws ONLY the detection/track box
  (+ minimal track state); the FC draws flight OSD. `overlay.py` should not
  duplicate battery/attitude/mode HUD — trim it to target marking only.

## 3. DEAD-END — CPU detection on the Zero 2W is not a flight detector

> **STATUS: RESOLVED.** The CPU detector stack has been **removed**. The IMX500
> AI camera is the only flight camera and detector (on-sensor inference, ~0 host
> CPU). Dev/sim hosts use light detectors (`color`, `haar`, ArUco) on synthetic,
> file, or webcam sources.

History: an earlier CPU detector measured ~221 ms (~4 Hz) on the Zero 2W. 4 Hz
detection of a maneuvering target is "a slideshow, not a tracking system," and no
amount of core-pinning or detect-period tuning made that path shippable. The
**IMX500 sensor-offload path is the architecturally correct cheap path**
(detection at sensor frame rate, ~0 host CPU — the IMX500/Hailo pattern), so the
CPU path was dropped entirely.

## 4. STRUCTURAL — control law deficiencies (fixable on Pi-class compute)

> **STATUS: ADDRESSED in code.** Alpha-beta target filter
> (`track/target_filter.py`) wired into the Pipeline ahead of the servo.
> Servo uses P + velocity FEEDFORWARD (`yaw_ff_gain`) — structural pursuit
> lag cancelled. Approach is now CLOSURE-REGULATED: forward lean ∝
> (filtered apparent size − desired hold size), so the aircraft eases off as
> it nears and noses up to back off if too close — the constant-speed
> collision risk is gone. Gains (`desired_bbox_frac`, `closure_p_gain`)
> still need flight tuning; size-as-range is a monotone proxy, not metric
> range — adequate for the closure loop, not for absolute standoff.

- **No closure-rate regulation.** `forward = const` whenever locked, no range
  term → drives at constant speed into the subject, no deceleration. Every
  production follow/pursuit system regulates closure by estimated range.
  (Subsumed by the §1 redesign — the attitude/rate redesign must handle
  closure explicitly; cheap proxy = bbox size.)
- **No target-velocity feedforward.** Pure-P against a moving target has a
  structural steady-state lag proportional to target speed (type-0 vs ramp).
  The use case *is* a moving target. Fix is feedforward from an estimated
  target image velocity, not a D term (D on a 3 Hz noisy pixel signal is
  worse).
- **Appearance-only gap-filler.** The detect-3Hz/control-30Hz split is sound
  *if* the gap-filler is kinematic (Kalman/alpha-beta carrying constant-
  velocity state, predicting through detector latency). MOSSE/KCF is an
  *appearance* tracker — no motion model, no velocity output, can't predict.
  The pattern's missing piece.

The convergent cheap fix for all three: a **2-state alpha-beta (or 4-state
constant-velocity Kalman) filter on the target centroid** — predicts through
detector latency, supplies the feedforward term, smooths deadband chatter.
Trivial cost. Plus bbox-size closure regulation.

## 5. SAFETY — the "confidently wrong" failure mode is unmitigated

> **STATUS: ADDRESSED in code.** The alpha-beta filter collapses track
> `quality` on the three "confidently wrong" failure modes — implausible
> centroid jumps (misdetection teleport, rejected as an outlier instead of
> acted on), class flips (locked a person, detector now says chair), and
> confidence decay / long coasting. The safety gate has a 5th gate:
> `quality < min_track_quality` -> muted ("low track quality"). The pilot's
> flight-mode switch remains the ultimate authority (§1). Still
> bench-validate the quality thresholds against real misdetection rates.

The safety gate handles *absence* of a target (switch off, disarmed, no
target, stale target). It has **zero mitigation for a *wrong* target**:

- Detector high-confidence misclassification → servo confidently flies the
  aircraft at the wrong object.
- Tracker drifts onto background texture → still "has a target," not stale,
  servo confidently commands toward garbage.
- No track-quality / confidence-decay gate.
- No plausibility bound on target motion (a detection that teleports across
  the frame should be rejected as implausible, not acted on).
- No max-engagement-duration or re-confirm; switch held = continuous
  commands toward whatever the tracker reports.
- Deadband-edge limit cycle (servo oscillates when error parks on the
  deadband boundary while the rate clamp saturates).

For a system whose entire purpose is "fly toward what the camera sees,"
confidently-wrong is the dominant hazard and the architecture currently does
nothing about it. Mitigations: track-quality gating (reject low-confidence /
high-IoU-drift), motion-plausibility bound (reject implausible jumps), an
alpha-beta innovation gate (reject detections far from prediction), and the
pilot-owned momentary mode switch as the ultimate authority (§1).

## 6. LATENT — camera-mount sign conventions

> **STATUS: ADDRESSED (code + procedure).** `guidance.yaw_sign`/`pitch_sign`
> (±1, applied in the servo) let an operator correct an inverted mount
> without code changes; `camera.hflip`/`vflip` un-mirror before detection.
> The mandatory bench self-test (which can't be done in software) is
> documented as `deployment-safety.md` §4 — must be re-run after any camera/
> lens/flip change.

`climb = -Kp·dy` and body-frame yaw-rate are correct **only** for a level,
forward-looking, non-mirrored camera. Any fixed downtilt or gimbal pitch
silently biases/inverts the vertical loop; a mirrored/flipped camera image
inverts yaw sign → divergent positive feedback ("drone spins away from
target"). Need an explicit camera-mount rotation and a sign self-test.

---

## Scorecard

**Sound (keep):**
- Loosely-coupled companion-over-UART topology (this *is* how the cheap tier
  is built — Skynode S, TFL-1, PrincipLoT).
- Analog link choice for this drone class (analog FPV is resurgent, not dead).
- The guidance *intent* + momentary-switch + FC-authoritative-failsafe
  philosophy (the principle is right; the control surface and switch wiring
  are wrong — see §1).
- IMX500 sensor-offload as the detection path.

**Wrong / must change:**
- ArduPilot velocity-in-GUIDED → must be `GUIDED_NOGPS` + `SET_ATTITUDE_TARGET`
  (FOUNDATIONAL, §1).
- Arming switch as Pi-side gate → must be the pilot's flight-mode channel (§1).
- (§2 retracted — Pi-as-camera + CVBS→FC-cam-pad is correct; only trim
  `overlay.py` to target-marking, drop duplicate flight-OSD elements.)
- CPU detection presented as a flight mode → removed; IMX500 is the camera (§3).
- Naive P servo → add alpha-beta target filter + feedforward + closure
  regulation (§4).
- No wrong-target safety → add track-quality + motion-plausibility gating (§5).

## The pivot

The project is roughly **one architectural pivot** from correct, and most of
the work is salvageable:

- KEEP: IMX500 detector path, the safety/guidance philosophy, the ArduPilot
  backend *concept*, the tracker/pipeline/config/test scaffolding, the
  Betaflight path as explicitly demo-only.
- CHANGE: control surface (attitude/rate via GUIDED_NOGPS); switch → pilot's
  flight-mode channel; CPU detection removed (IMX500 is the camera); add target
  state-estimator + closure + wrong-target gating; trim `overlay.py` to
  target-marking only.
- KEEP (corrected): the CVBS/DRM video subsystem IS the flight video path —
  Pi-as-camera is the design. (The MJPEG-over-HTTP browser preview has since
  been removed — output is IMX500 + analog composite / TV out only.)

The load-bearing redesign is now just ONE external interface: the FC control
surface (velocity-in-GUIDED → attitude-rate-in-GUIDED_NOGPS) plus the
switch-as-mode-channel wiring. The video path is sound as built. This is
still a deliberate decision, not an incremental patch — but smaller than the
original audit implied.
