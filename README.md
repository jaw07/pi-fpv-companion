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
- Monitors a designated RC channel. **While the pilot holds the switch active**,
  the Pi converts the tracked target into a **body-rate + thrust command** and
  sends it to the FC. **Switch off (STANDBY) → the Pi injects nothing that
  touches flight control** and the pilot has unmodified manual control (see
  *Failsafe principles*).

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
DRM KMS, `disable-bt` to free the PL011 UART), and wiring notes.

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
                 │  tracker        │   (multi-IoU association + operator select; flight default)
                 │     │           │
                 │     ▼           │
                 │  target filter  │   (alpha-beta: smooth, velocity, reject bad jumps)
                 │     │           │
                 │     ├──▶ overlay ──▶ framebuffer ──▶ CVBS pad ──▶ FC cam-in ──▶ VTX
                 │     │           │
                 │     ▼           │
                 │  visual servo   │   (pixel offset + closure → body-rate intent)
                 │     │           │
                 │     ▼           │
                 │  safety gate    │   (switch + armed + freshness + track quality + STANDBY contract)
                 │     │           │
                 │     ▼           │
                 │  fc backend     │   ArduPilot MAVLink  or  Betaflight MSP
                 +-----│-----------+
                       │ UART (TXD0/RXD0)
                       ▼
                  Flight Controller
```

## Camera

The IMX500 AI camera runs the detector **on the sensor NPU** and emits boxes via
picamera2 metadata at ~0 host-CPU cost; the Pi does IoU association across dense
detections, keyed to one target ID. Dev/sim hosts have no IMX500, so they use a
`SyntheticCamera`, `FileCamera`, or `WebcamCamera` with a light detector
(`color`, `haar`, ArUco) — the Gazebo SITL sim uses ColorBlob.

The default flight model is a **YOLO11n fine-tuned on VisDrone (aerial drone-view)
@416** — our own model (`models/imx500_network_yolo11n_visdrone416_pp.rpk`),
shipped in the repo and uploaded to the sensor at startup (~8 s "network firmware
upload"). It runs ~12.9 Hz on the NPU and is a **vehicle specialist** (strong on
car/van/truck/bus, weaker on pedestrians); `classes_of_interest`
(`pedestrian, people, car, van, truck, bus, motor`) and `conf_threshold` are
resolved against the model's own VisDrone label set. The stock COCO
`imx500_network_yolo11n_416_pp.rpk` ships alongside as an alternative. `detector.type`
is `none` on the flight config — inference is on-sensor, ncnn is intentionally not
installed.

The `Camera` interface yields `(frame, detections)`. The detector module is a
no-op when the camera (the IMX500) already produced detections, and runs inline
on a dev source that did not. The IMX500 is what flies.

### Tracker choice

**Multi-target IoU selection** (`tracker.type: multi_iou`, the IMX500 flight
default): the sensor emits detections every frame, so instead of a heavy
single-object tracker we track *every* detection with a stable id (the HUD shows
them all in STANDBY), and a momentary RC channel (`fc.select_channel`, ch9)
cycles which one is locked. **The selection is sticky** — held across frames and
across the mode switch — so the operator picks a target in STANDBY and it stays
locked through TRACK and DIVE; it never auto-swaps to the highest-confidence box.
Guidance can also **lead** a moving target (`guidance.lead_time_s`).

For dev sources that yield one box (synthetic/file/webcam), a classical OpenCV
tracker is available (`tracker.cv2_backend`). Benchmarked on the Zero 2W:
**MOSSE** runs ~1.2 ms/frame (the cheap default), KCF (~23 ms) is ~20× slower for
no gain here, CSRT (~199 ms, ~5 FPS) is confirmed unusable on this SoC. On the
IMX500 path none of these run — IoU association replaces them.

## Two FC backends

| Backend    | Protocol  | Control surface                                  | Switch read   |
|------------|-----------|--------------------------------------------------|---------------|
| ArduPilot  | MAVLink   | **`SET_ATTITUDE_TARGET` body rates + thrust in GUIDED_NOGPS** (`control_mode: guided_nogps`, the flight path). `stabilize`/`althold` RC-override (AETR sticks) are fallbacks. Handover/failsafe per `docs/deployment-safety.md`. | `RC_CHANNELS` |
| Betaflight | MSP       | `MSP_SET_RAW_RC` stick override in ANGLE mode (**demo only** — BF failsafe keys off RX, not MSP) | `MSP_RC`      |

The guidance layer emits one backend-agnostic intent and each backend translates.
On the flight path the visual servo produces **body rates + real thrust**
(`guidance/rate_control.py` → `backend.send_body_rates`).

TRACK follows and holds range; DIVE commits and moves altitude onto the target —
**closing onto a target below, level, or above**. Because the camera is bolted to
the airframe, pitch couples forward-closure with vertical aim, so DIVE uses a
forward lean for closure plus a pursuit-guidance thrust law (velocity vector onto
the line of sight, tracked on `VFR_HUD.climb`) to hold the target's frame
position. Holding a fixed frame point is a constant bearing → a collision course,
so the flight path follows the line of sight regardless of target altitude.
FOV-retention and dive geometry are verified by a closed-loop simulator and SITL.
See `docs/dive-guidance.md`.

**GPS-denied control (why body rates):** velocity setpoints
(`SET_POSITION_TARGET_LOCAL_NED`) need an EKF position solution a GPS-denied quad
lacks. Both GPS-free alternatives are supported: **GUIDED_NOGPS + `SET_ATTITUDE_TARGET`
body rates** (the flight path — smooth, dive-capable), and **RC-override into a
self-levelling pilot mode** (`stabilize`/`althold` fallbacks, where releasing the
override hands control straight back). STABILIZE allows a real dive (SITL 4.6:
~16 m/s / 77° vs ALT_HOLD's ~1–5 m/s) with no baro altitude floor. See
`docs/gps-denied-modes.md`.

**`control_mode: guided_nogps` (body-rate — THE FLIGHT PATH):** GUIDED_NOGPS with
`SET_ATTITUDE_TARGET` **body rates** + real thrust. Rates are integrated by the
airframe so the motion is smooth; the descent uses pursuit guidance. It
**requires** `GUID_OPTIONS` bit 3 (=8, ThrustAsThrust) — the preflight param check
sets+verifies it, otherwise the FC reads the thrust field as a climb-rate and the
dive planes.

**Engage (`fc.auto_guided`):** the pilot's FC-mode channel (ch6) selects the
flight mode; ch7 is the STANDBY/TRACK/DIVE engage; ch9 cycles the locked target.
With **`auto_guided: true`** (current default), flipping ch7 to TRACK/DIVE
**auto-commands the FC into GUIDED_NOGPS** (saving the prior mode and restoring it
on disengage), so ch7 alone enters control. With `auto_guided: false` the
companion never commands the mode — it stays released until the pilot puts the FC
in GUIDED_NOGPS (the `control_ready()` interlock). **Either way, the ch6 TX
flight-mode switch is the instant manual override / mid-dive abort.** Validated in
SITL + Gazebo camera-in-the-loop and HIL on a real FC (see *Status*) — **not yet
flight-validated**.

## Failsafe principles

1. **Pi is muted unless the switch is active.** No flight-control command flows
   until the pilot deliberately engages ch7.
2. **STANDBY injects nothing that touches flight control — in every FC mode.**
   This is the headline post-flight-2 change. In steady-state STANDBY the
   companion sends **no** non-zero `RC_CHANNELS_OVERRIDE`, **no**
   `SET_ATTITUDE_TARGET`, and **no** `DO_SET_MODE`. The only STANDBY wire traffic
   is the GCS heartbeat + telemetry stream requests, plus a short **all-zero
   "hand back to pilot" release burst** at the disengage edge, after which it goes
   radio-silent. It is verified by `safety_contract.py` (a pure contract checker)
   and proven on the real FC via `scripts/hil_standby_check.py`.
3. **Disarmed → never transmits control, in any switch state.** (Flight 2's
   self-launch came from a hover-thrust attitude command sent while disarmed; that
   path is closed.)
4. **The pilot always has instant manual recovery, independent of the Pi.** On the
   guided_nogps flight path, flipping ch6 to any pilot mode (STABILIZE) takes
   control back at once — the companion sees the FC is no longer in GUIDED_NOGPS
   (`control_ready()` false) and commands nothing. Disengaging ch7 to STANDBY
   while still in GUIDED_NOGPS leaves the FC to hold natively via its
   `GUID_TIMEOUT` (=3 s, auto-enforced) — the companion does **not** send a
   level-hover; it goes silent. (Accepted tradeoff: a ch7 disengage *mid-dive*
   coasts on the dive attitude up to `GUID_TIMEOUT`; mid-dive aborts go through ch6,
   not ch7.)
5. **Pi-death is caught by the FC.** The GCS heartbeat runs on its **own thread**
   (not tied to the frame loop, so a camera stall/watchdog restart can't gap it),
   arming the FC's GCS failsafe; `FS_GCS_TIMEOUT=20` and `FS_OPTIONS` bit 4 (=16,
   continue-in-pilot-modes) are auto-enforced so a Pi death only acts in
   GUIDED_NOGPS and never LANDs a manually-flown craft. If the **main loop itself
   wedges**, the companion deliberately **withholds** heartbeats so FS_GCS *can*
   fire. Set `FS_GCS_ENABLE=LAND` for GPS-denied flight.
6. **Pi self-mutes on a bad track.** Off-frame, stale, low confidence, *or* low
   filter quality (implausible jump / class flip → quality decays) → commands stop.
7. **Closure-rate limiting.** Forward pitch is regulated toward a target frame
   fraction and reverses past it — a collision guard, not a ram-the-target gain.

Pre-flight bench procedure (GUIDED_NOGPS arming + GUID_OPTIONS, ch7 mode / ch9
select, STANDBY contract + manual-recovery handover, camera-mount sign self-test,
wrong-target tuning, GPS-denied FS_GCS) is in `docs/deployment-safety.md`. Do not
fly without that.

## Status

Runs end-to-end **on the actual Pi Zero 2W** (service active, live IMX500,
composite/DRM output) and on the Mac dev host, from one codebase: the same
`Pipeline` runs with SyntheticCamera + UDP-loopback fake FC + viewer on the Mac,
and IMX500Camera + real UART + DRM framebuffer on the Pi, by swapping injected
components. **326 tests green** on both Mac and Pi (aarch64).

Validation state, honestly:
- Pipeline, perf, camera path, safety gating: **measured on real hardware**.
- **STANDBY safety contract: proven on a real flight controller.**
  `scripts/hil_standby_check.py` runs the production pipeline + real
  ArduPilotBackend (115200) + synthetic camera and wraps outbound sends: both
  natural-STANDBY and forced-engage-while-not-in-GUIDED (disarmed) inject zero
  flight-control traffic. The flight-2 fix, validated at the hardware level. The
  real-FC preflight param enforcement (`GUID_OPTIONS=8`, `FS_GCS_TIMEOUT=20`,
  `FS_OPTIONS=16`) is also confirmed written and read back on hardware.
- **Flight path — guided_nogps (`SET_ATTITUDE_TARGET` body rates):** SITL +
  **Gazebo camera-in-the-loop** validated — clean TRACK→DIVE→impact at 25/40/55 m,
  a moving target, STANDBY safe-hold, and a Pi-death hold, via
  `scripts/sitl_gz_validate.py` running the production `Pipeline` against SITL.
- Fallbacks (STABILIZE/ALT_HOLD + `RC_CHANNELS_OVERRIDE`, GPS-denied): SITL-validated
  on ArduCopter 4.6.3 — `validate_sitl.py`, `probe_nogps_modes.py`,
  `measure_dive_sitl.py` (STABILIZE ~16 m/s vs ALT_HOLD ~1–5).
- **Not yet flight-validated:** the guidance has never controlled a real aircraft
  in the air. Flight-tuning of gains + the props-off bench checklist are
  hardware-gated, tracked in `docs/deployment-safety.md`.

Setup (one-shot):

```
bash scripts/setup-venv.sh
```

Creates `.venv`, installs the package editable, and resolves the
`opencv-python` vs `opencv-contrib-python` conflict (a plain `opencv-python`
pulled in transitively silently overrides `cv2.legacy` trackers; the script
force-reinstalls the contrib package last so its `cv2.so` wins).

Deploy to a fresh Pi — from inside the cloned repo, one line does everything
(apt deps + `imx500-all` firmware, sync to `/opt`, venv, systemd unit enabled,
persistent journald, wifi-selfheal timer, and the boot config — free the UART via
`disable-bt` + composite TV-out), then reboot:

```
sudo -v && printf 'y\ny\n' | bash scripts/install-pi.sh && sudo reboot
```

`sudo -v` caches credentials so they don't collide with the script's two `[y/N]`
prompts; `printf 'y\ny\n'` answers them (yes to `imx500-all`, yes to the boot
config). Drop the `printf` pipe to confirm each prompt interactively. The service
is **enabled but not started** by the installer — it comes up on the post-install
reboot. See `scripts/install-pi.sh` and `scripts/setup-pi-boot.sh`.

Run the production entry point:

```
# On the Pi: IMX500 detections + composited feed (bbox + HUD) out the analog composite / TV out.
.venv/bin/python -m pi_fpv_companion --config config/imx500.yaml
# On a dev laptop there is no framebuffer/TV out — run headless:
.venv/bin/python -m pi_fpv_companion --config config/mac-dev.yaml --no-gui
```

A bench override `--force-mode {standby,track,dive}` (+ `--duration N`,
`--pi-scale`) exercises TRACK/DIVE without a bound RC TX. Dev verification is the
test suite: `.venv/bin/python -m pytest`. For SITL, see `docs/sitl.md`.

## Pi resource budget

Every tick is tracked against the Zero 2W's actual ceiling, not the Mac's:

- **Tick budget**: 33 ms (30 FPS).
- **RAM budget**: 200 MB of ~416 MB usable under Debian Trixie Lite.
- **Mac → Pi scaling**: workload-dependent; **measured, not extrapolated**.

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
| **MOSSE**   | **1.2 ms**| **822 FPS**      | Cheap dev default. No scale; detector re-seeds |
| MedianFlow  | 11.6 ms   | 86 FPS           | Fails on fast motion                 |
| KCF         | 22.7 ms   | 44 FPS           | ~20× slower than MOSSE for no gain here |
| CSRT        | 199 ms    | 5 FPS            | Confirmed unusable on this SoC       |

## Can the Pi handle it?

**Yes.** The IMX500 sensor NPU runs the detector at ~0 ms host cost, so the Pi's
job is passthrough, IoU association, filter, overlay, and MAVLink — measured at
0.4 ms p50 on the Zero 2W. 30 FPS with large headroom.

## What's built

- Core types: `Detection`, `FilteredTarget`, `GuidanceIntent`, `RateIntent`,
  `SwitchState`; `HOVER_THRUST`, `ZERO_INTENT`.
- Visual servo — two paths: the **guided_nogps body-rate** law
  (`guidance/rate_control.py`: yaw/roll blend, pursuit-guidance dive thrust,
  impact latch) and the legacy attitude/RC-override servo (pixel offset + closure,
  deadzone, clamps, `yaw_sign`/`pitch_sign` for mount orientation).
- Alpha-beta target filter — 4-state CV filter; smooths, estimates velocity,
  rejects implausible jumps and class flips by decaying quality, coasts through
  measurement gaps.
- Safety gate + **STANDBY command contract** (`safety_contract.py`) — switch /
  armed / freshness / min track quality, plus the hard "STANDBY injects nothing"
  contract with a pure verifier and a `scripts/hil_standby_check.py` HIL harness
  and `scripts/check_wire_contract.py` tlog/live reader.
- `FlightController` Protocol with two backends:
  - ArduPilot — real MAVLink I/O; `SET_ATTITUDE_TARGET` body rates in GUIDED_NOGPS
    (flight path) and `RC_CHANNELS_OVERRIDE` AETR in STABILIZE/ALT_HOLD (fallback);
    own-thread GCS heartbeat; preflight `GUID_OPTIONS`/`FS_GCS_TIMEOUT`/`FS_OPTIONS`
    enforcement; `auto_guided` ch7 mode-engage. SITL- + HIL-validated.
  - Betaflight — real MSP v1 I/O (encoder/decoder + state machine), validated
    against a loopback-serial fake. Stick override is demo-only.
- `Camera` / `Detector` / `Tracker` Protocols with: `IMX500Camera`
  (on-sensor NPU — the flight camera, VisDrone YOLO11n default), `SyntheticCamera`,
  `FileCamera`, `WebcamCamera` (dev/sim); `ColorBlobDetector`, `HaarFaceDetector`,
  ArUco (light dev detectors); `ClassicalCv2Tracker`, `IouAssociator` (multi-target
  select).
- Video out: `overlay`, `LinuxFramebuffer` (`/dev/fb0`), `DrmFramebuffer`
  (DRM dumb-buffer for Trixie default KMS) — flight is the analog composite / TV out.
- Companion flight recorder (`flight_log.py` → `var/flight/*.jsonl`, 10 Hz
  decision trail), persistent journald, wifi-selfheal timer.
- `config.py` (typed YAML loader), `main.py` (component factory by config),
  `perf.py` (Pi-budget verdict).
- systemd unit + `scripts/install-pi.sh` / `scripts/setup-pi-boot.sh`.
- Docs: `docs/how-it-works.md`, `docs/user-guide.md`,
  `docs/architecture-audit.md` (durable audit record),
  `docs/deployment-safety.md` (pre-flight bench checklist), `docs/dive-guidance.md`,
  `docs/gps-denied-modes.md`, `docs/ardupilot-vertical-control-research.md`,
  `docs/sitl.md`, `docs/hardware.md`, `docs/pi-hardware-todos.md`.

## Next

1. ~~SITL + Gazebo validation of the GPS-denied control surface~~ — **done**.
2. ~~HIL bench validation of the STANDBY safety contract on a real FC~~ — **done**
   (`scripts/hil_standby_check.py`, real ArduPilotBackend; preflight params
   confirmed on hardware).
3. **Camera bring-up on the airframe** as the flight detector (IMX500 + VisDrone
   model) — in progress.
4. Validate ch7 behaviour from FC dataflash logs (RCIN.C7 + MODE) before relying
   on `auto_guided`'s ch7 mode-engage in flight.
5. Flight-tuning the guided_nogps gains on the airframe with prop guards and a
   kill switch — the guidance has never controlled a real aircraft in the air.
