# pi-fpv-companion

Onboard computer-vision companion for an analog-FPV drone, running on a
Raspberry Pi Zero 2W mounted on the airframe. **The Pi is the camera.** A Sony
IMX500 AI camera does detection on-sensor; the Pi draws a target box and pushes that out
its composite (CVBS) pad into the flight controller's camera-in. The FC overlays
its own flight OSD and feeds the analog VTX to the goggles. Separately, the Pi
talks to the FC over UART and — only while the pilot holds an RC switch — sends
guidance so the drone flies toward the tracked target.

## What it does

- Reads frames from a Sony **IMX500 AI camera** (on-sensor inference); dev/sim
  hosts use synthetic, file, or webcam sources instead.
- Detects, filters, and locks onto a target (alpha-beta state filter rejects
  implausible jumps and class flips, and supplies a velocity estimate).
- Draws a bounding box + minimal state onto the video and pushes it out the
  Pi's composite pad to the FC's camera-in. The FC owns the flight OSD; the Pi
  deliberately draws almost nothing else.
- Talks to the flight controller over UART (MAVLink for ArduPilot, MSP for
  Betaflight).
- Monitors a designated RC channel. While the pilot holds the switch active,
  the Pi converts the tracked target into an **attitude command** (roll/pitch/
  yaw-rate/thrust) and sends it to the FC. Switch off → the Pi is silent on the
  control surface and the pilot has unmodified manual control.

## Hardware

- Raspberry Pi Zero 2W (BCM2710A1, quad-A53 @ 1 GHz, 512 MB RAM; ~416 MB usable
  under Debian Trixie Lite)
- Sony IMX500 AI Camera — inference runs on the sensor NPU; the Pi receives
  boxes via picamera2 metadata. **This is the flight camera and detector.**
- Flight controller running ArduPilot **or** Betaflight, with an analog camera
  input that gets forwarded to the analog VTX
- 5V BEC sized for Pi peaks (>1 A headroom) — separate from the VTX rail if
  possible

See `docs/hardware.md` for solder points, `config.txt` settings (composite/PAL,
DRM KMS), and wiring notes.

## Architecture

```
                       Pi Zero 2W
                  +-----------------+
   IMX500 ─────▶  │  capture        │
                 │     │           │
                 │     ▼           │
                 │  detector       │   (IMX500 on-sensor metadata — flight; light dev detectors on the Mac)
                 │     │           │
                 │     ▼           │
                 │  tracker        │   (MOSSE between detector refreshes; default)
                 │     │           │
                 │     ▼           │
                 │  target filter  │   (alpha-beta: smooth, velocity, reject bad jumps)
                 │     │           │
                 │     ├──▶ overlay ──▶ framebuffer ──▶ CVBS pad ──▶ FC cam-in ──▶ VTX
                 │     │           │
                 │     ▼           │
                 │  visual servo   │   (pixel offset + closure → attitude intent)
                 │     │           │
                 │     ▼           │
                 │  safety gate    │   (switch + armed + freshness + track quality)
                 │     │           │
                 │     ▼           │
                 │  fc backend     │   ArduPilot MAVLink  or  Betaflight MSP
                 +-----│-----------+
                       │ UART (TXD0/RXD0)
                       ▼
                  Flight Controller
```

## Camera

The IMX500 AI camera runs an on-sensor SSD-MobileNetV2 (COCO-trained) detector
and emits boxes via picamera2 metadata at ~0 host-CPU cost; the Pi does IoU/Kalman
association across dense detections, keyed to one target ID. Dev/sim hosts have no
IMX500, so they use a `SyntheticCamera`, `FileCamera`, or `WebcamCamera` with a
light detector (`color`, `haar`, ArUco) — the Gazebo SITL sim uses ColorBlob.

The `Camera` interface yields `(frame, detections)`. The detector module is a
no-op when the camera (the IMX500) already produced detections, and runs inline
on a dev source that did not. The IMX500 is what flies.

### Tracker choice (MOSSE default)

OpenCV's classical trackers, benchmarked on the Zero 2W (see perf tables):
**MOSSE** runs ~1.2 ms/frame and is the default — it has no scale adaptation,
but the detector re-seeds it on every burst, which is exactly what covers
MOSSE's weakness. KCF (~23 ms) is ~20× slower for no useful gain here. CSRT
(~199 ms, ~5 FPS) is confirmed unusable on this SoC; it remains available as a
config flip (`tracker.cv2_backend: csrt`) for slow targets on a faster board
only. On the IMX500 path the sensor emits detections every frame, so a heavy
single-object tracker is wasted work — we use IoU association instead.

**Multi-target selection** (`tracker.type: multi_iou`, the IMX500 default):
tracks *every* detection with a stable id (the HUD shows them all in STANDBY),
and a momentary RC channel (`fc.select_channel`) cycles which one is locked. The
selection is held across frames and across the mode switch, so the operator picks
a target in STANDBY and it stays locked through TRACK and DIVE. Guidance can also
**lead** a moving target (`guidance.lead_time_s`) — aim at the intercept rather
than tail-chase.

## Two FC backends

| Backend    | Protocol  | Control surface                                  | Switch read   |
|------------|-----------|--------------------------------------------------|---------------|
| ArduPilot  | MAVLink   | **`SET_ATTITUDE_TARGET` body rates + thrust in GUIDED_NOGPS** (`control_mode: guided_nogps`, the flight path). `stabilize`/`althold` RC-override (AETR sticks) are fallbacks. Handover/failsafe per `docs/deployment-safety.md`. | `RC_CHANNELS` |
| Betaflight | MSP       | `MSP_SET_RAW_RC` stick override in ANGLE mode (**demo only** — BF failsafe keys off RX, not MSP) | `MSP_RC`      |

The guidance layer emits one backend-agnostic intent —
`GuidanceIntent(roll_deg, pitch_deg, yaw_rate_dps, thrust, timestamp)` — and
each backend translates. Thrust `0.5` ≈ hold (hover throttle in STABILIZE; baro
hold in ALT_HOLD); the Pi steers yaw, forward pitch, and (for DIVE) descent.

TRACK follows and holds range; DIVE commits and moves altitude onto the target —
**closing onto a target below, level, or above**. Because the camera is bolted to
the airframe, pitch couples forward-closure with vertical aim, so DIVE uses a
forward lean for closure (steep/fast onto a below target, gentle when climbing to
an above one) and a commanded vertical **rate** (tracked on `VFR_HUD.climb`) to
hold the target's frame position. Holding a fixed frame point
is a constant bearing → a collision course, so the flight path follows the line of
sight regardless of target altitude. FOV-retention and dive geometry are verified
by a closed-loop simulator (`tests/closed_loop_sim.py`, `scripts/sim_track_dive.py`)
and SITL. See `docs/dive-guidance.md`.

**GPS-denied control (why body rates):** velocity setpoints
(`SET_POSITION_TARGET_LOCAL_NED`) need an EKF position solution a GPS-denied quad
lacks. Both GPS-free alternatives are supported: **GUIDED_NOGPS + `SET_ATTITUDE_TARGET`
body rates** (the flight path — smooth, dive-capable, see below), and **RC-override
into a self-levelling pilot mode** (`stabilize`/`althold` fallbacks, where releasing
the override hands control straight back). STABILIZE allows a real dive (SITL 4.6:
~16 m/s / 77° vs ALT_HOLD's ~1–5 m/s) with no baro altitude floor (the companion
owns altitude). See `docs/gps-denied-modes.md`

**`control_mode: guided_nogps` (body-rate — THE FLIGHT PATH):** GUIDED_NOGPS with
`SET_ATTITUDE_TARGET` **body rates** + real thrust (`guidance/rate_control.py`,
`backend.send_body_rates`). Rates are integrated by the airframe so the motion is
smooth; the descent uses pursuit guidance (velocity vector onto the line of sight).
It **requires** `GUID_OPTIONS` bit 3 (ThrustAsThrust) — the preflight param check
sets+verifies it, otherwise the FC reads the thrust field as a climb-rate and the
dive planes. The pilot's FC-mode channel selects GUIDED_NOGPS; ch7 is the
STANDBY/TRACK/DIVE engage; ch9 cycles the locked target. Validated in SITL +
Gazebo camera-in-the-loop (clean TRACK→DIVE→impact at 25/40/55 m, a moving target,
STANDBY safe-hold, Pi-death hold) — **not yet hardware-validated**. `stabilize` /
`althold` (RC-override) remain as fallbacks. See `docs/gps-denied-modes.md` and
`docs/architecture-audit.md` §1.

## Failsafe principles

1. **Pi is muted unless the switch is active.** No commands flow until the
   pilot deliberately engages the RC channel.
2. **The pilot always has instant manual recovery, independent of the Pi.** On the
   guided_nogps flight path the pilot's FC-mode channel selects GUIDED_NOGPS to let
   the companion command; flipping it to any pilot mode (STABILIZE) takes control
   back at once — the companion sees the FC is no longer in GUIDED_NOGPS
   (`control_ready()` false) and commands nothing. Disengaging the ch7 switch to
   STANDBY (while still in GUIDED_NOGPS) holds a level hover; it never leaves the FC
   coasting on the last attitude. A dead Pi → the FC holds via its GUIDED command
   timeout, and the companion's ~1 Hz GCS heartbeat arms FS_GCS as a backstop (set
   it to LAND for GPS-denied). On the `stabilize`/`althold` fallbacks the model is
   RC-override instead: the Pi injects AETR only while engaged and releases them in
   STANDBY, and a dead Pi reverts via ArduPilot's RC-override timeout.
3. **Pi self-mutes on a bad track.** Off-frame, stale, low confidence, *or* low
   filter quality (implausible jump / class flip → quality decays) → commands
   stop. This is the wrong-target guard.
4. **FC owns its own failsafe.** If the Pi dies, the FC's MAVLink/MSP timeout
   failsafe handles the silence — never rely solely on Pi-side watchdogs.
5. **Closure-rate limiting.** Forward pitch is regulated by bbox size toward a
   target frame fraction and reverses past it — a collision guard, not a
   ram-the-target gain.

Pre-flight bench procedure (GUIDED_NOGPS arming + GUID_OPTIONS, ch7 mode / ch9
select, STANDBY hover-hold + manual-recovery handover, camera-mount sign self-test,
wrong-target tuning, GPS-denied FS_GCS) is in `docs/deployment-safety.md`. Do not
fly without that.

## Status

Runs end-to-end **on the actual Pi Zero 2W** (service active, live IMX500,
composite/DRM output) and on the Mac dev host, from one codebase: the same
`Pipeline` runs with SyntheticCamera + UDP-loopback fake FC + viewer on the Mac,
and IMX500Camera + real UART + DRM framebuffer on the Pi, by swapping injected
components. **273 tests green** on both Mac and Pi (aarch64).

Validation state, honestly:
- Pipeline, perf, camera path, safety gating: **measured on real hardware** (see
  tables below).
- **Flight path — guided_nogps (`SET_ATTITUDE_TARGET` body rates):** SITL +
  **Gazebo camera-in-the-loop** validated. Clean TRACK→DIVE→impact at 25/40/55 m
  and a moving target; `scripts/sitl_gz_validate.py` runs the production `Pipeline`
  against SITL and confirms the GUID_OPTIONS bit-3 preflight (readback=8), the
  STANDBY level-hover safe-hold (ΔAGL +0.2 m), and a Pi-death hold (ΔAGL +0.0 m, no
  tumble). See `docs/gps-denied-modes.md`, `docs/architecture-audit.md` §1.
- Fallbacks (STABILIZE/ALT_HOLD + `RC_CHANNELS_OVERRIDE`, GPS-denied): SITL-validated
  on ArduCopter 4.6.3 — `validate_sitl.py` 9/9, `probe_nogps_modes.py` 10/10,
  `measure_dive_sitl.py` (STABILIZE ~16 m/s vs ALT_HOLD ~1–5).
- **Not yet hardware-validated** in flight: the guidance has never controlled a real
  aircraft. Flight-tuning of gains + the props-off bench checklist are
  hardware-gated, tracked in `docs/deployment-safety.md`.

Setup (one-shot):

```
bash scripts/setup-venv.sh
```

Creates `.venv`, installs the package editable, and resolves the
`opencv-python` vs `opencv-contrib-python` conflict (a plain `opencv-python`
pulled in transitively silently overrides `cv2.legacy.TrackerMOSSE`; the
script force-reinstalls the contrib package last so its `cv2.so` wins).

Run the production entry point:

```
# On the Pi: IMX500 detections + composited feed (bbox + HUD) out the analog composite / TV out.
.venv/bin/python -m pi_fpv_companion --config config/imx500.yaml
# On a dev laptop there is no framebuffer/TV out — run headless:
.venv/bin/python -m pi_fpv_companion --config config/mac-dev.yaml --no-gui
```

Dev verification is the test suite (no on-screen viewer): `.venv/bin/python -m pytest`.
For SITL, see `docs/sitl.md` (`scripts/validate_sitl.py`, `scripts/fly_sitl.py`).

## Pi resource budget

Every tick is tracked against the Zero 2W's actual ceiling, not the Mac's:

- **Tick budget**: 33 ms (30 FPS).
- **RAM budget**: 200 MB of ~416 MB usable under Debian Trixie Lite.
- **Mac → Pi scaling**: workload-dependent; **measured, not extrapolated** (the
  tables below are real Pi numbers).

`PerfMonitor` reports p50/p95/p99 tick latency + peak RSS and prints a Pi
verdict at the end of every demo.

| Workload                                        | Mac p50  | Pi p50  | Multiplier |
|-------------------------------------------------|----------|---------|------------|
| Pipeline scaffold (synth + IoU + MAVLink)       | 0.18 ms  | 0.40 ms | **2.2×**   |

The IMX500 detector runs on the sensor NPU, so it adds ~0 host-CPU cost — the
Pi's per-tick work is association, filter, overlay, and MAVLink.

Classical tracker benchmark @ 720×576 textured frames on Pi Zero 2W (measured):

| Tracker     | p50 (Pi)  | FPS sustainable | Notes                                |
|-------------|-----------|------------------|--------------------------------------|
| **MOSSE**   | **1.2 ms**| **822 FPS**      | Default. No scale, but detector re-seeds |
| MedianFlow  | 11.6 ms   | 86 FPS           | Fails on fast motion                 |
| KCF         | 22.7 ms   | 44 FPS           | ~20× slower than MOSSE for no gain here |
| CSRT        | 199 ms    | 5 FPS            | Confirmed unusable on this SoC       |

## Can the Pi handle it?

**Yes.** The IMX500 sensor NPU runs the detector at ~0 ms host cost, so the Pi's
job is passthrough, IoU association, filter, overlay, and MAVLink — measured at
0.4 ms p50 on the Zero 2W. 30 FPS with large headroom.

## What's built

- Core types: `Detection`, `FilteredTarget`, `GuidanceIntent` (attitude domain),
  `SwitchState`; `HOVER_THRUST`, `ZERO_INTENT`.
- Visual servo — pixel offset + velocity feedforward → yaw rate; bbox-size
  closure → forward pitch; deadzone, clamps, `yaw_sign`/`pitch_sign` for mount
  orientation.
- Alpha-beta target filter — 4-state CV filter; smooths, estimates velocity,
  rejects implausible jumps and class flips by decaying quality (no snapping),
  coasts on prediction through measurement gaps.
- Safety gate — switch / armed / freshness / min track quality.
- `FlightController` Protocol with two backends:
  - ArduPilot — real MAVLink I/O, `RC_CHANNELS_OVERRIDE` AETR sticks in STABILIZE
    (default) or ALT_HOLD, GPS-denied; SITL-validated on ArduCopter 4.6.3.
  - Betaflight — real MSP v1 I/O (encoder/decoder + state machine), validated
    against a loopback-serial fake. Stick override is demo-only.
- `Camera` / `Detector` / `Tracker` Protocols with: `IMX500Camera`
  (on-sensor NPU — the flight camera), `SyntheticCamera`, `FileCamera`,
  `WebcamCamera` (dev/sim); `ColorBlobDetector`, `HaarFaceDetector`, ArUco
  (light dev detectors); `ClassicalCv2Tracker` (MOSSE/KCF/CSRT/MedianFlow),
  `IouAssociator`.
- Video out: `overlay`, `LinuxFramebuffer` (`/dev/fb0`), `DrmFramebuffer`
  (DRM dumb-buffer for Trixie default KMS) — flight is the analog composite / TV out.
- `config.py` (typed YAML loader), `main.py` (component factory by config),
  `perf.py` (Pi-budget verdict).
- systemd unit + `scripts/install-pi.sh` / `scripts/setup-pi-boot.sh`.
- Docs: `docs/architecture-audit.md` (durable audit record, §1–§6
  ADDRESSED/RETRACTED), `docs/deployment-safety.md` (pre-flight bench
  checklist), `docs/sitl.md`, `docs/hardware.md`, `docs/pi-hardware-todos.md`.

## Next

1. ~~SITL + Gazebo validation of the GPS-denied control surface~~ — **done**:
   guided_nogps (the flight path) flies clean TRACK→DIVE→impact in Gazebo
   camera-in-the-loop with GUID_OPTIONS preflight, STANDBY safe-hold, and Pi-death
   hold confirmed against SITL (`scripts/sitl_gz_validate.py`); RC-override fallbacks
   SITL-validated on 4.6.3.
2. Hardware-in-the-loop bench tests per `docs/deployment-safety.md` (GUIDED_NOGPS
   arming + GUID_OPTIONS, ch7/ch9, STANDBY hover-hold + FC-mode manual recovery,
   camera-sign self-test, GPS-denied FS_GCS) — **next gate**.
3. Re-run `validate_sitl.py` against the exact flight firmware (4.0.3 is older
   than a likely flight build; EKF3 defaults / minor `SET_ATTITUDE_TARGET`
   handling differ on Copter 4.5/4.6).
4. IMX500 camera bring-up as the flight detector.
5. Flight-tuning servo gains on the airframe with prop guards and a kill switch.
