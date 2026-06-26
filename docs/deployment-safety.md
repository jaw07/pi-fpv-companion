# Deployment & flight-safety checklist

The code-side audit items are addressed (see `architecture-audit.md`). The
items here CANNOT be enforced in software — they are wiring, FC-parameter, and
bench-validation requirements. **Do not fly until every box is checked.**

This is a guidance-injection system that flies the aircraft toward what a
camera sees. The dominant hazards are (a) the aircraft not returning to the
pilot, and (b) confidently flying at the wrong thing. Both have layered
mitigations below; none is sufficient alone.

---

## 1. Engage switch + handover — guided_nogps flight path (audit §1, guidance.md)

The flight path is **GUIDED_NOGPS + `SET_ATTITUDE_TARGET`** (body rates + real thrust):
`fc.control_mode: guided_nogps`. The pilot's flight-MODE channel selects GUIDED_NOGPS to
hand the airframe to the companion; **ch7** (engage) selects STANDBY / TRACK / DIVE; **ch9**
cycles the locked target among detections. The companion never changes flight mode itself.

Handover / failsafe (validated in SITL + Gazebo via `scripts/sitl_gz_validate.py`):
- **Manual recovery — always available, independent of the Pi:** the pilot flips the FC-mode
  channel OUT of GUIDED_NOGPS (e.g. to STABILIZE). The companion sees the FC is no longer in
  GUIDED_NOGPS (`control_ready()` false) and commands nothing → instant manual control.
- **STANDBY (ch7) injects NOTHING, in every FC mode** (operator requirement). With the FC
  still in GUIDED_NOGPS, ArduCopter's own `GUID_TIMEOUT` (3 s, auto-enforced) levels and
  holds zero climb natively after the last engaged setpoint. **Consequence, by explicit
  choice:** disengaging mid-dive leaves the FC on the dive attitude for up to that 3 s
  timeout — flip the FC mode out of GUIDED_NOGPS (ch6) for instant recovery, and prefer
  ch6 (not ch7) as the mid-dive abort.
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
- [ ] Bench-verified (props off): ch7 → STANDBY with the FC in GUIDED_NOGPS sends NOTHING
      (zero SET_ATTITUDE_TARGET); ~3 s after the last engaged setpoint the FC's GUID_TIMEOUT
      hold reports level + zero climb on its own.
- [ ] Bench-verified: killing the Pi (`sudo systemctl stop pi-fpv-companion`) → the craft holds
      (GUIDED timeout / FS_GCS) and the pilot's FC-mode flip recovers it.

## 2. GUIDED_NOGPS flight params (audit §1, guidance.md)

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
- [ ] **`FS_OPTIONS` bit 4 (=16)** (auto-enforced): GCS failsafe is ignored in pilot-controlled
      modes. Without it, a Pi death/restart LANDs a craft being flown MANUALLY on the sticks
      (suspected flight-2 mechanism). The failsafe still LANDs in GUIDED_NOGPS, where losing
      the companion means nobody is flying.
- [ ] **`FS_GCS_TIMEOUT` ≥ 20 s** (enforced from `config/imx500.yaml` at startup). The heartbeat
      runs on its own thread (independent of the camera), but a camera-watchdog **process restart**
      still gaps it for a few seconds — at the 5 s default that LANDs the craft on every camera
      hiccup (flight-2 finding, 2026-06-12). 20 s rides through a restart; a truly dead Pi and the
      systemd rung-4 reboot (~48 s) still fail safe. While ENGAGED, the GUIDED command timeout
      (~3 s, levels the craft) covers the gap — FS_GCS is the outer backstop, not the first line.
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

---

# Hardware validation (real Pi Zero 2W)

Every Mac measurement in this project is an extrapolation. When the Pi arrives,
work this list and update the numbers in README + `perf.py`.

## Failure model (decided)

Camera vs detector faults are handled differently, on purpose:

- **Camera dead** (no frames, or never delivered a first frame): FATAL. No
  video = the pilot's primary flight reference is gone = unrecoverable
  in-process. `frames()` raises after `_STALE_FRAME_TIMEOUT_S`; systemd
  restarts the unit (re-inits the camera — brief black, then back).
- **Detection dead** (sensor model fault, corrupt detections): NON-FATAL.
  Video still flows; only AI tracking is lost. The tracker loses its target,
  the safety gate zeroes intent — pilot keeps manual control with video +
  overlay live. A bug in the AI must never black out the pilot's feed.

This reverses the 2nd-audit "raise on detection failure" fix for the *detection*
path only (silent-death was the original bug; fatal-death killed the video;
loud-but-non-fatal is correct). Camera path stays fatal.

## Performance

- [x] **systemd service validated** — installed, started as `User=copilot`
      (in `video` group, so DRM works without sudo), killed mid-run and
      observed clean restart in 5 s. Fixed deprecated `StartLimitIntervalSec`
      → `StartLimitInterval`. Journal logging captures stdout cleanly.
- [ ] **Detect cadence vs target speed (empirical)**. The IMX500 emits
      detections at sensor frame rate (~30 Hz). Fly a slow target (~5 m/s
      relative) and a fast one (~15 m/s relative). Measure lock-loss rate.
- [x] **Classical tracker FPS at 720×576** measured. KCF 44 FPS, MOSSE 822 FPS,
      MedianFlow 86 FPS, CSRT 5 FPS. **Default switched to MOSSE** — its
      no-scale weakness is fully compensated by IMX500 detections every frame.
- [ ] **Full pipeline tick** on IMX500 frames (capture + IoU + overlay + fb +
      MAVLink — detection is on-sensor). Should be comfortably under 10 ms.
- [ ] **Peak RSS** of the full pipeline. Must stay under 200 MB (out of ~350
      MB usable after Bookworm Lite).
- [ ] **CPU thermal throttling** after 10+ min sustained load. A53 throttles
      under heat; if observed, add a heatsink to the SoC or accept the lower
      sustained frame rate.

## CVBS / framebuffer

- [x] **Trixie composite config corrected** — research showed `enable_tvout=1`,
      `sdtv_mode`, `sdtv_aspect` are all ignored under default vc4-kms-v3d.
      Removed them. Added `dtoverlay=vc4-kms-v3d,composite` to config.txt and
      `vc4.tv_norm=PAL` to cmdline.txt. Will activate after next reboot.
- [x] **DRM dumb-buffer framebuffer implemented** in `video/drm_framebuffer.py`.
      Pure ctypes wrapper around the 7 DRM ioctls we need — no libdrm Python
      binding required. Single-buffered (no page-flip), restores fbcon on
      shutdown. `main.py:_build_sink` falls through fb0→DRM→cv2.imshow
      automatically. Kernel ioctls validated against the actual Pi (struct
      sizes + GETRESOURCES + GETCONNECTOR all work). Will bind to the
      `Composite-1` connector once the next reboot brings it up.
- [ ] **First-light test after soldering TV pads**: reboot Pi with the new
      composite config, plug any analog RCA monitor into the TV pad + GND,
      check for the kernel framebuffer console / login prompt appearing.
      That's the dumbest possible "is composite working at all" test, no
      pi-fpv-companion code involved.
- [ ] **Confirm `/dev/fb0` format once enabled**. Read
      `/sys/class/graphics/fb0/virtual_size`, `bits_per_pixel`, `stride`.
      Code assumes 720×576 RGB565 (PAL); NTSC would be 720×480.
- [ ] **CVBS signal quality** on the FC's analog input. Solder the TV pads,
      route a short coax to the FC cam-in, view through the analog VTX into
      goggles. Note any visible noise, banding, sync issues.
- [ ] **Glass-to-glass latency** from IMX500 → CVBS out → FC → VTX → goggles.
      Goal under 100 ms. Measure with a physical "clap" + frame counting.

## UART

- [ ] **115200 baud reliability** under sustained MAVLink traffic. ArduPilot
      sends HEARTBEAT + RC_CHANNELS + many other streams. Use `mavlink_inspector`
      or pymavlink to verify no dropped messages over 5 min.
- [x] **Pi UART config applied** on Trixie. `dtoverlay=disable-bt` set,
      `serial-getty@ttyAMA0`/`@ttyS0` disabled, `console=serial0,...` removed
      from cmdline. After reboot `/dev/serial0 -> ttyAMA0` (PL011). Ready
      for a real FC.
- [ ] **Switch read latency**. Toggle RC channel 7 on the radio; measure time
      until backend's `read_switch()` reflects the change. Should be <100 ms.

## Camera

- [ ] **Camera capture-thread efficiency** (deferred from audit). The latest-
      wins capture thread runs `capture_array()` at sensor rate, doing a full
      BGR conversion + ~1.2 MB alloc every frame. If the pipeline consumes fewer
      frames than the sensor produces, those are wasted conversions on a
      CPU-limited Zero 2W. Safe fix needs restructuring to the request/`make_array`
      API with explicit buffer release — defer until the pipeline is otherwise
      stable.

## Camera audit (done — applied)

- [x] **AeExposureMode=Short** — was uncapped (AE could run 20 ms+ exposures,
      heavy motion blur on a moving drone). Now biased to fast shutter.
      Config-driven (`camera.exposure_mode`).
- [x] **NoiseReductionMode=Fast** — was Off (grainy, hurts detector). Now
      fast denoise, negligible latency. Config-driven (`camera.noise_reduction`).
- [x] **Capture-thread error handling** — a libcamera fault used to silently
      kill the thread and stall the pipeline with a frozen frame (silent
      in-flight failure). Now records the error and `frames()` raises loud
      after 10 consecutive failures.
- [x] **hflip/vflip config-driven** — set per camera mounting orientation
      without a code edit (`camera.hflip` / `camera.vflip`).
- [x] **ScalerCrop maximized** — picamera2 default was a tight ~47%-width
      centre crop; now the widest aspect-correct sensor region (max FOV the
      lens allows). FOV beyond the lens is fixed by the IMX500's optics.
- [ ] **IMX500 detection rate + metadata format**. `picamera2.imx500.IMX500`
      gives a `CnnOutputTensor` or similar — verify field name and parse pattern
      match current picamera2 docs.
- [ ] **IMX500 model upload time**. Loading the `.rpk` to the sensor on startup
      takes a few seconds; the systemd unit should account for it.

## Power

- [ ] **Brownout under load**. 1A+ peak draw on a small drone BEC during boot
      + Wi-Fi + camera. Add the 470 µF bulk cap as documented in `docs/hardware.md`.
- [ ] **Undervoltage warnings** in `dmesg`. Watch for `Under-voltage detected`
      and adjust BEC if needed.

## Behavior

- [ ] **Failsafe gate timing**. Toggle ch7 to STANDBY; in guided_nogps the
      companion should command a level hover (on the fallbacks, intent → zero /
      release). Measure round-trip time. Also confirm leaving GUIDED_NOGPS hands
      the sticks back at once.
- [ ] **FC failsafe on Pi crash**. Kill the Pi process mid-flight in SITL; the
      FC's GUIDED command timeout must take over (and `FS_GCS` → LAND fire once the
      ~1 Hz GCS heartbeat stops) within its configured window.
- [ ] **GUIDED_NOGPS behavior with stale commands**. ArduCopter is conservative
      about stale `SET_ATTITUDE_TARGET` setpoints. Verify our command rate is high
      enough to avoid the FC reverting to hover, and that `GUID_OPTIONS` bit 3
      (ThrustAsThrust) reads back set so thrust is real throttle (not a climb-rate).
