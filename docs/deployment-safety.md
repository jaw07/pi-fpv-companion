# Deployment & flight-safety checklist

The code-side audit items are addressed (see `architecture-audit.md`). The
items here CANNOT be enforced in software — they are wiring, FC-parameter, and
bench-validation requirements. **Do not fly until every box is checked.**

This is a guidance-injection system that flies the aircraft toward what a
camera sees. The dominant hazards are (a) the aircraft not returning to the
pilot, and (b) confidently flying at the wrong thing. Both have layered
mitigations below; none is sufficient alone.

---

## 1. Engage switch + pilot stays in the matching mode (audit §1, gps-denied-modes.md)

The control path is **RC_CHANNELS_OVERRIDE into a self-levelling pilot mode**
(default **STABILIZE**; ALT_HOLD selectable) — not GUIDED_NOGPS. The pilot keeps
the craft in that mode; while the engage switch is in TRACK/DIVE the Pi overrides
the AETR sticks, and in STANDBY it **releases** them (`release()` sends override
0 = "use the receiver"). Because it's a pilot mode, handback is immediate and
needs no flight-mode change — and a Pi crash is *fail-safe*: when override frames
stop, ArduPilot's RC-override timeout reverts the channels to the real radio.

`fc.control_mode` MUST match the FC's flight mode (stabilize ↔ STABILIZE,
althold ↔ ALT_HOLD) or the throttle mapping is wrong.

Required wiring / config:
- [ ] The "engage" switch is a spare 3-position channel; `fc.switch_channel`
      points at it (STANDBY / TRACK / DIVE via track/dive thresholds).
- [ ] The pilot's flight-MODE channel selects the mode matching `fc.control_mode`
      (STABILIZE by default). The Pi never changes flight mode.
- [ ] The Pi overrides only the AETR channels (RCMAP roll/pitch/throttle/yaw,
      default 1–4); it releases 5–8 so the pilot keeps mode + engage.
- [ ] Bench-verified: flipping the engage switch to STANDBY instantly returns
      full manual stick authority (override released).
- [ ] Bench-verified: killing the Pi process (`sudo systemctl stop
      pi-fpv-companion`) mid-engagement — within the FC's RC-override timeout
      (`RC_OVERRIDE_TIME`, ~1–3 s) the radio sticks resume automatically.

## 2. ArduPilot RC-override flight (audit §1, gps-denied-modes.md)

A bare FPV quad has no GPS / EKF position estimate. The backend injects AETR
sticks via `RC_CHANNELS_OVERRIDE` — needs only baro + IMU, no GPS, no EKF origin
(SITL-proven on 4.6.3). No SET_ATTITUDE_TARGET, no GUIDED, no `GUID_OPTIONS`.

- [ ] FC firmware: ArduCopter 4.6+. On boot the companion **validates + writes**
      the FC params it needs — `ANGLE_MAX` (= `fc.angle_max_deg`, so commanded lean
      = actual lean) and the companion RC channels' `*_OPTION = 0` — verifying each
      write. **Check the startup log** (`ok`/`set`/`write-fail` per param). It does
      NOT touch serial/baud (the link it's on) or flight modes/failsafes — set
      those yourself. Disable with `fc.enforce_params_on_start: false`.
- [ ] **`control_mode: stabilize` (default) has no FC altitude hold** — throttle
      is direct. TRACK altitude is held by the companion's **adaptive hover**
      (a vertical-velocity PI loop on `VFR_HUD.climb`: it learns the hover throttle
      and damps climb, SITL-proven to level out from a wrong seed in ~6 s). So set
      `stab_hover_throttle_us` to a *rough* seed only — the learner refines it.
      **Requires `VFR_HUD` streamed from the FC** (SR*_EXTRA2); without it, adaptive
      hover falls back to the fixed seed. There is NO altitude *floor*: a commanded
      DIVE descends until released (SITL: ~16 m/s, 77° path) — intentional for a
      diving craft. Fly with altitude margin and a finger on the engage switch.
      Bench-verify the learner holds level before trusting it.
- [ ] **Closed-loop DIVE vertical-rate sign** (`stabilize` only) — the dive's
      throttle loop tracks a commanded climb rate on `VFR_HUD.climb`. A reversed
      throttle channel or inverted climb sign makes the loop **diverge** (commands
      descent → climbs → commands more descent → flyaway). This is as dangerous as
      a yaw-sign inversion. **Validate in SITL before flight**:
      `scripts/validate_vrate_sitl.py` must show a commanded **−3 m/s descends**
      and **+2 m/s climbs** (not reversed). The bench check (props off) cannot test
      this — it needs a hover; confirm in SITL or a cautious tethered/altitude-
      margin hover with a finger on the engage switch. **Also requires `VFR_HUD`**;
      without it the dive falls back to an open-loop throttle map (still descends,
      but uncalibrated).
- [ ] `control_mode: althold` alternative: throttle = climb rate (0.5 = hold via
      baro), descent capped at `PILOT_SPEED_DN`. Safer altitude, gentle dive only.
- [ ] **Stick signs** (`rc_roll_sign` / `rc_pitch_sign` / `rc_yaw_sign`):
      bench/SITL-validate per §4. Defaults (+1,+1,+1) are SITL-correct but
      TX/RCMAP/airframe dependent.
- [ ] The vehicle arms in the chosen mode and responds to test sticks (props off)
      in the correct directions.
- [ ] `RC_OVERRIDE_TIME` (or equivalent) set so a Pi stall reverts to the radio
      quickly. FC's own RC-loss + battery failsafes configured/tested.

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

- [ ] Props OFF, FC in the matching mode (STABILIZE default), engaged (Pi overriding).
- [ ] Place a target to the **right** of frame centre. Confirm the FC
      commands **yaw right** (nose rotates toward the target), not away.
      If it yaws away → flip `guidance.yaw_sign` (or fix `camera.hflip`).
- [ ] Move the target **closer** (larger in frame). Confirm pitch eases
      toward zero / noses up (backs off), does not accelerate forward.
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
