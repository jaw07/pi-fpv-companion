# Deployment & flight-safety checklist

The code-side audit items are addressed (see `architecture-audit.md`). The
items here CANNOT be enforced in software — they are wiring, FC-parameter, and
bench-validation requirements. **Do not fly until every box is checked.**

This is a guidance-injection system that flies the aircraft toward what a
camera sees. The dominant hazards are (a) the aircraft not returning to the
pilot, and (b) confidently flying at the wrong thing. Both have layered
mitigations below; none is sufficient alone.

---

## 1. Engage switch + handover — guided_nogps flight path (audit §1, gps-denied-modes.md)

The flight path is **GUIDED_NOGPS + `SET_ATTITUDE_TARGET`** (body rates + real thrust):
`fc.control_mode: guided_nogps`. The pilot's flight-MODE channel selects GUIDED_NOGPS to
hand the airframe to the companion; **ch7** (engage) selects STANDBY / TRACK / DIVE; **ch9**
cycles the locked target among detections. The companion never changes flight mode itself.

Handover / failsafe (validated in SITL + Gazebo via `scripts/sitl_gz_validate.py`):
- **Manual recovery — always available, independent of the Pi:** the pilot flips the FC-mode
  channel OUT of GUIDED_NOGPS (e.g. to STABILIZE). The companion sees the FC is no longer in
  GUIDED_NOGPS (`control_ready()` false) and commands nothing → instant manual control.
- **STANDBY (ch7) while still in GUIDED_NOGPS:** the companion HOLDS a level hover (self-trimming
  to null climb). It never leaves the FC coasting on the last (possibly dive) attitude.
- **Pi death:** with no `SET_ATTITUDE_TARGET`, ArduCopter's GUIDED command timeout holds the
  craft; the companion's ~1 Hz GCS heartbeat also arms FS_GCS (§2). Recovery is the FC-mode flip.

Required wiring / config:
- [ ] `fc.switch_channel: 7` — spare 3-position channel (STANDBY / TRACK / DIVE via the
      track/dive thresholds).
- [ ] `fc.select_channel: 9` — spare channel; a rising edge past 1700 µs **in STANDBY** cycles
      the locked target (needs `tracker.type: multi_iou`). Frozen once engaged.
- [ ] The pilot's flight-MODE channel can select **GUIDED_NOGPS** (engage) *and* a pilot mode
      like STABILIZE (manual recovery). Bench-verify the mode switch reaches both.
- [ ] Bench-verified: flipping the FC mode OUT of GUIDED_NOGPS returns full manual stick authority.
- [ ] Bench-verified (props off): ch7 → STANDBY with the FC in GUIDED_NOGPS commands a LEVEL
      hover (level attitude + hover thrust), not a dive attitude.
- [ ] Bench-verified: killing the Pi (`sudo systemctl stop pi-fpv-companion`) → the craft holds
      (GUIDED timeout / FS_GCS) and the pilot's FC-mode flip recovers it.

## 2. GUIDED_NOGPS flight params (audit §1, gps-denied-modes.md)

A bare FPV quad has no GPS / EKF position estimate. GUIDED_NOGPS + `SET_ATTITUDE_TARGET`
(body rates + thrust) needs only baro + IMU — no GPS, no EKF origin (SITL-proven on 4.6.3).

- [ ] FC firmware: ArduCopter 4.6+. On boot the companion **validates + writes** the params it
      needs — `ANGLE_MAX`, the companion RC channels' `*_OPTION = 0`, and **`GUID_OPTIONS` bit 3
      (ThrustAsThrust)** (OR-ed in, verified by readback). **Check the startup log**
      (`ok`/`set`/`write-fail`; the `GUID_OPTIONS(ThrustAsThrust)` line must report success).
      It does NOT touch serial/baud or the failsafe params — set those yourself.
      Disable with `fc.enforce_params_on_start: false`.
- [ ] **`GUID_OPTIONS` bit 3 is MANDATORY.** Without it the FC reads the thrust field as a
      climb-rate (0.5 = hold altitude), so "throttle 0" never descends and the dive planes.
      SITL-confirmed the preflight sets it (readback = 8).
- [ ] **GPS-denied GCS failsafe = LAND.** Set `FS_GCS_ENABLE`/`FS_OPTIONS` so a lost GCS lands
      (NOT RTL/SmartRTL — they need GPS). The companion emits a ~1 Hz GCS heartbeat so FS_GCS is
      armed on Pi death; the GUIDED command timeout is the inner backstop. Configure + bench-test.
- [ ] **Arming GPS-denied:** the EKF must allow arming without GPS (e.g. `EK3_SRC1_POSXY`/`VELXY`
      = 0, `POSZ` = Baro; relax `ARMING_CHECK` only on the bench). The craft must reach
      GUIDED_NOGPS armed + airborne (e.g. take off in a pilot mode, then switch to GUIDED_NOGPS).
- [ ] **No altitude floor:** a commanded DIVE descends to impact; the impact latch cuts throttle
      at ground contact (AGL keyed to the disarmed-captured home). Fly with margin; finger on ch7.
- [ ] **Hover thrust is learned in flight** (TWR-independent — a high-TWR quad hovers well below
      0.5): the rate path trims hover toward null climb during TRACK and the STANDBY hover-hold.
      **Requires `VFR_HUD` streamed** (SR*_EXTRA2).

**Fallbacks — `stabilize` / `althold` (RC-override path).** Set `fc.control_mode` to match the
FC's flight mode. These inject AETR via `RC_CHANNELS_OVERRIDE` into a self-levelling pilot mode;
STANDBY releases the override (instant handback), and a dead Pi reverts via `RC_OVERRIDE_TIME`.
STABILIZE = direct throttle + the companion's adaptive-hover loop (no FC alt hold, true ~16 m/s
dive); ALT_HOLD = climb-rate throttle (gentle). On the stabilize path the **closed-loop DIVE
vertical-rate sign** must be SITL-validated (`scripts/validate_vrate_sitl.py`: −3 m/s descends,
+2 m/s climbs) — a reversed sign diverges into a flyaway. Stick signs (`rc_*_sign`) and arming +
test sticks (props off) bench/SITL-validate per §4.

## 3. Betaflight path is DEMO-ONLY (audit §1)

Betaflight failsafe keys off the **RX link, not MSP**. A hung Pi at full
`MSP_SET_RAW_RC` override with a healthy RX does **not** failsafe — frozen
sticks. Do not fly the Betaflight backend as a guidance system.

- [ ] If used at all: `msp_override_channels` leaves throttle + arm on the
      physical RX. Never override arm. Treat as a ground/demo tool only.

## 4. Camera-mount sign self-test (audit §6)

`yaw_rate` (body) and `pitch` are correct only for a **level, forward-looking,
non-mirrored** camera. A mirrored/flipped image inverts the yaw sign →
divergent positive feedback: the aircraft spins/leans *away* from the target,
faster and faster. This is the most dangerous silent misconfiguration.

Mitigations in code: `camera.hflip`/`camera.vflip` (un-mirror before detection)
and `guidance.yaw_sign`/`guidance.pitch_sign` (±1 operator override, applied in
the servo). Neither helps if set wrong — they must be **bench-validated**:

- [ ] Props OFF, FC in **GUIDED_NOGPS**, ch7 engaged to TRACK (companion commanding body rates).
- [ ] Place a target to the **right** of frame centre. Confirm the commanded **yaw rate is to
      the right** (nose rotates toward the target), not away. If it yaws away → flip
      `guidance.yaw_sign` (or fix `camera.hflip`).
- [ ] In DIVE, a target **below** frame centre must command **nose-down** (pitch toward the
      target), not nose-up. Reversed → flip `guidance.pitch_sign` (or fix `camera.vflip`).
- [ ] Re-confirm after ANY change to camera mounting, lens, or hflip/vflip.

## 5. Wrong-target / track-quality (audit §5)

The alpha-beta filter collapses track quality on implausible jumps, class
flips, and confidence decay; the safety gate mutes below
`safety.min_track_quality`.

- [ ] Bench-tune `min_track_quality` against the real detector's false-lock
      rate in the actual operating environment (lighting, clutter).
- [ ] Confirm: occlude / swap the target — guidance mutes within the
      watchdog window, overlay box goes grey, no command toward the wrong
      object.

## 6. Detector path (audit §3)

- [ ] Flight detector is **IMX500** (`camera.type: imx500`) — on-sensor
      inference, ~30 FPS, ~0 host CPU. It is the only flight camera/detector.
- [ ] Dev/sim hosts use light detectors (`color`, `haar`, ArUco) on synthetic,
      file, or webcam sources. Never a flight config.

## 7. General

- [ ] First flights: open area, observers clear, low/slow, finger on the
      mode switch, short engagements.
- [ ] Bench-validate the full chain (camera → detect → track → servo →
      FC) with props OFF before any powered test.
