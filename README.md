# pi-fpv-companion

Onboard computer-vision companion for an analog-FPV drone, running on a
Raspberry Pi Zero 2W mounted on the airframe. **The Pi is the camera.** A CSI
sensor feeds onboard detection; the Pi draws a target box and pushes that out
its composite (CVBS) pad into the flight controller's camera-in. The FC overlays
its own flight OSD and feeds the analog VTX to the goggles. Separately, the Pi
talks to the FC over UART and — only while the pilot holds an RC switch — sends
guidance so the drone flies toward the tracked target.

## What it does

- Reads frames from either a regular Pi Camera (CSI) or a Sony **IMX500 AI
  camera** (CSI, on-sensor inference).
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
- Camera, either:
  - Standard Pi Camera (any CSI module) — inference runs on the Pi CPU (NCNN).
    **Dev/sim only** — see "camera paths" below.
  - Sony IMX500 AI Camera — inference runs on the sensor NPU; the Pi receives
    boxes via picamera2 metadata. **This is the flight detector.**
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
   Pi Cam ────▶  │  capture        │
   or IMX500 ─▶  │     │           │
                 │     ▼           │
                 │  detector       │   (CPU-NCNN dev/sim, or IMX500 metadata — flight)
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

## Two camera paths

| Path     | Inference site | Pi CPU cost | Detect rate | Role |
|----------|----------------|-------------|-------------|------|
| IMX500   | Sensor NPU     | ~0          | Frame rate  | **Flight detector.** IoU/Kalman association across dense detections, keyed to one target ID |
| Pi Cam   | Pi (NCNN)      | High        | ~4 Hz       | **Dev/sim only.** NanoDet-Plus CPU inference is a slideshow on this SoC; MOSSE bridges between refreshes |

Both expose the same `Camera` interface: they yield `(frame, detections)`. The
detector module is a no-op when the camera already produced detections.

The CPU/NCNN path is not a flight detector — ~4 Hz refresh drifts on fast FPV
targets, and `main.py` prints a startup WARN whenever a CPU detector is
selected. It exists for no-hardware development and for higher-RAM SBCs
(Pi 4/CM4). The IMX500 is what flies. See `docs/architecture-audit.md` §3.

### Tracker choice (MOSSE default)

OpenCV's classical trackers, benchmarked on the Zero 2W (see perf tables):
**MOSSE** runs ~1.2 ms/frame and is the default — it has no scale adaptation,
but the detector re-seeds it on every burst, which is exactly what covers
MOSSE's weakness. KCF (~23 ms) is ~20× slower for no useful gain here. CSRT
(~199 ms, ~5 FPS) is confirmed unusable on this SoC; it remains available as a
config flip (`tracker.cv2_backend: csrt`) for slow targets on a faster board
only. On the IMX500 path the sensor emits detections every frame, so a heavy
single-object tracker is wasted work — we use IoU association instead.

## Two FC backends

| Backend    | Protocol  | Control surface                                  | Switch read   |
|------------|-----------|--------------------------------------------------|---------------|
| ArduPilot  | MAVLink   | `RC_CHANNELS_OVERRIDE` AETR sticks in **STABILIZE** (default; roll/pitch = lean angle, yaw = rate, throttle = direct; `control_mode: althold` for baro alt-hold). Releases to the pilot when disengaged. | `RC_CHANNELS` |
| Betaflight | MSP       | `MSP_SET_RAW_RC` stick override in ANGLE mode (**demo only** — BF failsafe keys off RX, not MSP) | `MSP_RC`      |

The guidance layer emits one backend-agnostic intent —
`GuidanceIntent(roll_deg, pitch_deg, yaw_rate_dps, thrust, timestamp)` — and
each backend translates. Thrust `0.5` ≈ hold (hover throttle in STABILIZE; baro
hold in ALT_HOLD); the Pi steers yaw, forward pitch, and (for DIVE) descent.

TRACK follows and holds range; DIVE commits and is **altitude-agnostic** — it
dives onto a target below, pursues one level ahead, and climbs toward one above,
keyed on the target's true line-of-sight elevation (FC attitude + in-frame
position). Because the camera is bolted to the airframe, every command rotates
the field of view; the FOV-retention and dive geometry are verified by a
closed-loop simulator (`tests/closed_loop_sim.py`, `scripts/sim_track_dive.py`).
See `docs/dive-guidance.md`.

**Why RC override into STABILIZE (GPS-denied):** velocity setpoints
(`SET_POSITION_TARGET_LOCAL_NED`) need an EKF position solution a GPS-denied quad
lacks. GUIDED_NOGPS + `SET_ATTITUDE_TARGET` (also GPS-free) was the first design;
it was replaced by injecting AETR sticks into a self-levelling **pilot** mode so
releasing the override hands control straight back (a dead Pi fail-safes via the
FC's RC-override timeout). **STABILIZE** is the default over ALT_HOLD because it
allows a real dive (SITL 4.6: ~16 m/s / 77° vs ALT_HOLD's ~1–5 m/s); the cost is
no baro altitude floor (the companion owns altitude). See `docs/gps-denied-modes.md`
and `docs/architecture-audit.md` §1.

## Failsafe principles

1. **Pi is muted unless the switch is active.** No commands flow until the
   pilot deliberately engages the RC channel.
2. **Engage releases to the pilot, not a mode lockout.** The pilot keeps the FC
   in a self-levelling mode (STABILIZE by default); the Pi overrides the AETR
   sticks only while engaged and releases them in STANDBY. Disengage → the radio
   sticks resume immediately; a dead Pi →
   ArduPilot's RC-override timeout reverts to the radio. Either way the pilot has
   instant manual recovery, independent of the Pi.
3. **Pi self-mutes on a bad track.** Off-frame, stale, low confidence, *or* low
   filter quality (implausible jump / class flip → quality decays) → commands
   stop. This is the wrong-target guard.
4. **FC owns its own failsafe.** If the Pi dies, the FC's MAVLink/MSP timeout
   failsafe handles the silence — never rely solely on Pi-side watchdogs.
5. **Closure-rate limiting.** Forward pitch is regulated by bbox size toward a
   target frame fraction and reverses past it — a collision guard, not a
   ram-the-target gain.

Pre-flight bench procedure (STABILIZE/ALT_HOLD arming + RC override, hover-throttle
+ sign self-test for a mirrored/rotated camera mount, Betaflight-demo caveat,
wrong-target tuning) is in `docs/deployment-safety.md`. Do not fly without that.

## Status

Runs end-to-end **on the actual Pi Zero 2W** (service active, live PiCam,
composite/DRM output) and on the Mac dev host, from one codebase: the same
`Pipeline` runs with SyntheticCamera + UDP-loopback fake FC + viewer on the Mac,
and PiCamCamera + real UART + DRM framebuffer on the Pi, by swapping injected
components. **171 tests green** on both Mac and Pi (aarch64).

Validation state, honestly:
- Pipeline, perf, camera path, safety gating: **measured on real hardware** (see
  tables below).
- Control surface (STABILIZE/ALT_HOLD + `RC_CHANNELS_OVERRIDE`, GPS-denied):
  **SITL-validated on ArduCopter 4.6.3** — `validate_sitl.py` 9/9 (RC-override
  steers yaw/pitch/roll in the correct sense, signs verified), `probe_nogps_modes.py`
  10/10 (STABILIZE + ALT_HOLD arm+steer with GPS fully disabled), and
  `measure_dive_sitl.py` quantifies the dive (STABILIZE ~16 m/s vs ALT_HOLD ~1–5).
  See `docs/gps-denied-modes.md`, `docs/sitl.md`, `docs/architecture-audit.md` §1.
- Flight-tuning of servo gains and all hardware-in-the-loop bench tests:
  hardware-gated, tracked in `docs/deployment-safety.md`.

Setup (one-shot):

```
bash scripts/setup-venv.sh
```

Creates `.venv`, installs the package editable, and resolves the
`ncnn → opencv-python` vs `opencv-contrib-python` conflict (ncnn pulls in the
non-contrib opencv which silently overrides `cv2.legacy.TrackerMOSSE`; the
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
| NanoDet-Plus-M @ 416 NCNN                       | 10.3 ms  | 586 ms  | **57×**    |
| NanoDet-Plus-M @ 320 NCNN                       | 6.7 ms   | 347 ms  | **52×**    |
| NanoDet-Plus-M @ 256 NCNN                       | —        | 221 ms  | —          |

Classical tracker benchmark @ 720×576 textured frames on Pi Zero 2W (measured):

| Tracker     | p50 (Pi)  | FPS sustainable | Notes                                |
|-------------|-----------|------------------|--------------------------------------|
| **MOSSE**   | **1.2 ms**| **822 FPS**      | Default. No scale, but detector re-seeds |
| MedianFlow  | 11.6 ms   | 86 FPS           | Fails on fast motion                 |
| KCF         | 22.7 ms   | 44 FPS           | ~20× slower than MOSSE for no gain here |
| CSRT        | 199 ms    | 5 FPS            | Confirmed unusable on this SoC       |

Full end-to-end pipeline on Pi (MOSSE + NanoDet @ 256 + detect every 7 frames,
real textured 720×576 frames, simulated 30 FPS arrival, sync vs async detector):

| Mode | p50 tick | p95 tick | max | Over-budget (>33 ms) |
|---|---|---|---|---|
| Sync (blocks main loop) | 0.1 ms | 217 ms | 230 ms | **14% of frames** |
| **Async worker thread (default)** | 0.1 ms | 2.3 ms | 2.7 ms | **0% of frames** |

The async detector moves the 221 ms inference off the critical path. Same
detector refresh rate (~4 Hz), but the main loop never stalls. The worker dying
is non-fatal — it must never take down the pilot's video.

### Real camera, real hardware (measured on IMX708 / Camera Module 3)

| Stage | Measured |
|---|---|
| `PiCamCamera` capture only @ 720×576 | 25.5 FPS |
| Full pipeline (cam → NanoDet@256 async → MOSSE → servo → safety → FC) | **14 FPS effective** |
| Main-loop tick p95 (async detector) | 2.7 ms |
| Peak RSS | 120 MB / 200 MB budget |

The 25→14 FPS drop is CPU contention: NCNN's inference threads compete with
libcamera's ISP and the main loop for 4 A53 cores. Mitigated by core-pinning
(`cpu.pin`: detector cores {2,3}, pipeline {0,1}). 14 Hz is fine for FPV — the
composite output scans at 50 Hz PAL regardless; only the overlay + guidance
update at 14 Hz. NanoDet produced stable detections on live frames (same object
±5 px across 80 frames), so tracker lock is clean — but this is still the
dev/sim path; flight uses the IMX500.

## Can the Pi handle it?

**Yes on the IMX500 path. The PiCam/NCNN path is dev/sim only (measured).**

- **IMX500 (flight)**: the sensor NPU runs the detector at ~0 ms host cost. The
  Pi's job is passthrough, IoU association, filter, overlay, MAVLink — measured
  at 0.4 ms p50 on the Zero 2W. 30 FPS with large headroom.
- **PiCam + NanoDet (dev/sim)**: NCNN inference is the floor — 221 ms at input
  256, ~4 Hz refresh with the async detector. Fine for slow follow in
  development; drifts on adversarial FPV targets, which is why it does not fly.
- **YOLOv8n is not viable here.** It OOM-reboots the Zero 2W; NanoDet-Plus is
  the only CPU detector that fits the RAM budget. `Yolov8Detector` exists as a
  drop-in for higher-RAM boards (Pi 4/CM4), not for the Zero 2W.

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
- `Camera` / `Detector` / `Tracker` Protocols with: `SyntheticCamera`,
  `FileCamera`, `WebcamCamera`, `PiCamCamera` (picamera2), `IMX500Camera`
  (on-sensor NPU); `ColorBlobDetector`, `HaarFaceDetector`, `NanoDetDetector`
  (NCNN + GFL decode), `Yolov8Detector` (higher-RAM boards only);
  `ClassicalCv2Tracker` (MOSSE/KCF/CSRT/MedianFlow), `IouAssociator`.
- `AsyncDetector` — detector on a pinned worker thread; non-fatal on death.
- Video out: `overlay`, `LinuxFramebuffer` (`/dev/fb0`), `DrmFramebuffer`
  (DRM dumb-buffer for Trixie default KMS) — flight is the analog composite / TV out.
- `cpu_affinity` — pins detector and pipeline to disjoint A53 core sets.
- `config.py` (typed YAML loader), `main.py` (component factory by config),
  `perf.py` (Pi-budget verdict).
- systemd unit + `scripts/install-pi.sh` / `scripts/setup-pi-boot.sh`.
- Docs: `docs/architecture-audit.md` (durable audit record, §1–§6
  ADDRESSED/RETRACTED), `docs/deployment-safety.md` (pre-flight bench
  checklist), `docs/sitl.md`, `docs/hardware.md`, `docs/pi-hardware-todos.md`.

## Next

1. ~~ArduPilot SITL validation of the GPS-denied control surface~~ — **done on
   ArduCopter 4.6.3**: RC override 9/9 control sense, probe 10/10 (STABILIZE +
   ALT_HOLD GPS-off), dive quantified (STABILIZE ~16 m/s). STABILIZE is the default.
2. Hardware-in-the-loop bench tests per `docs/deployment-safety.md` (FC in
   ALT_HOLD, RC-override handback, sign self-test, failsafe verification) — **next gate**.
3. Re-run `validate_sitl.py` against the exact flight firmware (4.0.3 is older
   than a likely flight build; EKF3 defaults / minor `SET_ATTITUDE_TARGET`
   handling differ on Copter 4.5/4.6).
4. IMX500 camera bring-up as the flight detector.
5. Flight-tuning servo gains on the airframe with prop guards and a kill switch.
