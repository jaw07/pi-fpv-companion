# Things to validate on real Pi Zero 2W hardware

Every Mac measurement in this project is an extrapolation. When the Pi arrives,
work this list and update the numbers in README + `perf.py`.

## Failure model (decided)

Camera vs detector faults are handled differently, on purpose:

- **Camera dead** (no frames, or never delivered a first frame): FATAL. No
  video = the pilot's primary flight reference is gone = unrecoverable
  in-process. `frames()` raises after `_STALE_FRAME_TIMEOUT_S`; systemd
  restarts the unit (re-inits the camera — brief black, then back).
- **Detector worker dead** (NCNN fault, model corruption, OOM in the worker):
  NON-FATAL. Video still flows; only AI tracking is lost. `poll()` returns
  None (logged once, loud), the tracker loses its target, the safety gate
  zeroes intent — pilot keeps manual control with video + overlay live. A
  bug in the AI must never black out the pilot's feed.

This reverses the 2nd-audit "poll() raises" fix for the *detector* path only
(silent-death was the original bug; fatal-death killed the video; loud-but-
non-fatal is correct). Camera path stays fatal.

## Detector bugs

- [x] **`Yolov8Detector` not viable on Pi Zero 2W**. Decoder was already broken
      (output decode mismatch), but the bigger blocker: **running YOLOv8 NCNN
      inference at any input size auto-reboots the Pi**, almost certainly OOM.
      YOLOv8n + activation tensors during inference exceeds the 416MB RAM
      ceiling. Decoder bug is now moot. NanoDet-Plus is the **only viable
      detector** on this SoC; YOLOv8 might still be useful on Pi 3B+ / Pi 4
      with more RAM but is incompatible with Zero 2W. Class kept in the
      codebase for those higher-RAM SBCs.

## Performance

- [x] **NCNN inference Mac→Pi multiplier**. **Measured 52-57× on Pi Zero 2W**
      (vs M-series Mac). Matches research extrapolation; `perf.py` docstring updated.
- [x] **NanoDet-Plus-M @ 320 latency**. **Measured 347 ms.** Default
      `detect_period_frames: 10` overruns by ~14 ms; should be 11+ at this input
      size, or switch to 256 input.
- [x] **NanoDet-Plus-M @ 256 latency**. **Measured 221 ms.** Workable with
      `detect_period_frames: 7` for 4.3 Hz refresh.
- [x] **NanoDet-Plus-M @ 416 latency**. **Measured 586 ms.** Too slow for
      practical use unless detect cadence is >18 frames (1.7 Hz).
- [x] **VisDrone NanoDet @ 416 latency: 383 ms** on Pi Zero 2W (smaller model,
      2.3MB vs 5MB COCO). 10 aerial classes (pedestrian/people/bicycle/car/
      van/truck/tricycle/awning/bus/motor). Test on city scene: 0 detections
      (expected — aerial training). **Locked at 416 input — the NCNN export
      is fixed-shape**, so we can't run smaller. To use at smaller input,
      re-export with `--dynamic` from the original training weights.
- [x] **COCO NanoDet model is dynamic-shape** — output anchor count scales
      with input (1360 @ 256, 2125 @ 320, 3598 @ 416). This is why our 256
      input works on the COCO model but not on the VisDrone one.
- [x] **systemd service validated** — installed, started as `User=copilot`
      (in `video` group, so DRM works without sudo), killed mid-run and
      observed clean restart in 5 s. Fixed deprecated `StartLimitIntervalSec`
      → `StartLimitInterval`. Journal logging captures stdout cleanly.
- [ ] **Detect cadence vs target speed (empirical)**. 4 Hz detect at 256 input
      is still borderline for fast targets per recent UAV CV literature.
      Fly a slow target (~5 m/s relative) and a fast one (~15 m/s relative).
      Measure lock-loss rate. Decide if threaded detector is necessary.
- [x] **Thread the detector**. Implemented (`AsyncDetector` worker, default-on
      in Pipeline). Measured on Pi: p95 tick dropped from 217 ms (sync) to 2.3 ms
      (async), 0% overruns past the 33 ms budget vs 14% with sync. Same detector
      refresh rate (~4 Hz); main loop no longer stalls.
- [ ] **YOLOv8n @ 256 latency**. Research says ~170–200 ms — verify and decide
      whether YOLO's broader class coverage beats NanoDet's smaller footprint.
- [x] **Classical tracker FPS at 720×576** measured. KCF 44 FPS, MOSSE 822 FPS,
      MedianFlow 86 FPS, CSRT 5 FPS. **Default switched to MOSSE** — its
      no-scale weakness is fully compensated by detector re-seed every cycle.
      End-to-end pipeline with MOSSE + NanoDet @ 256 = **29.7 FPS sustained**.
- [ ] **Full pipeline tick** on real PiCam frames (capture + detect + track +
      overlay + fb write + MAVLink). Must stay under 33 ms for 30 FPS or under
      50 ms for 20 FPS.
- [ ] **Full pipeline tick** on IMX500 frames (capture + IoU + overlay + fb +
      MAVLink, no CPU detector). Should be comfortably under 10 ms.
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
- [ ] **Glass-to-glass latency** from pi cam → CVBS out → FC → VTX → goggles.
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

- [x] **PiCam capture validated** on IMX708 / Camera Module 3. `PiCamCamera`
      captures real 720×576 BGR at 25.5 FPS (libcamera picks 1536×864 sensor
      mode, ISP downscales). Full pipeline with async NanoDet@256 + MOSSE =
      **14 FPS effective** (CPU contention: NCNN 2 threads vs camera ISP +
      main loop on 4 A53 cores). NanoDet detections stable ±5 px across 80
      live frames — clean tracker lock. Acceptable for FPV.
- [x] **NCNN worker core-pinning implemented + measured.** `cpu.pin: true`
      pins the detector worker (+ its NCNN pool threads) to cores {2,3} and
      the main process (capture + loop + output) to {0,1}. Measured on the
      flight path (DRM composite, no preview): p50 tick 91.7->31.5 ms, p95
      313->244 ms, content refresh ~7->~10.5 FPS, RSS 145->135 MB. ~10.5 FPS
      is the practical Zero 2W ceiling for CPU NanoDet; 30 FPS needs IMX500
      (sensor-side detection) or a faster SBC. `src/pi_fpv_companion/cpu_affinity.py`.
- [ ] **Camera capture-thread efficiency** (deferred from audit). The latest-
      wins capture thread runs `capture_array()` at sensor rate (~25 fps),
      doing a full BGR conversion + ~1.2 MB alloc every frame even though the
      pipeline consumes ~14 fps. ~11 wasted conversions/sec on a CPU-starved
      Zero 2W. Safe fix needs restructuring to the request/`make_array` API
      with explicit buffer release — defer until the pipeline is otherwise
      stable; matters less in flight (no JPEG-encode contention there).

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
      lens allows). FOV beyond the lens needs a wide CSI module (Cam Module 3
      Wide ~120°, or Arducam ultra-wide ~160°) — no analog cam input on a
      Pi Zero 2W.
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

- [ ] **Failsafe gate timing**. Toggle the switch off; the next-frame intent
      sent to the FC should be zero. Measure round-trip time.
- [ ] **FC failsafe on Pi crash**. Kill the Pi process mid-flight in SITL; the
      FC's GUIDED-mode timeout must take over within its configured window.
- [ ] **GUIDED-mode behavior with stale commands**. ArduCopter is conservative
      about stale velocity setpoints. Verify our 20 Hz command rate is high
      enough to avoid the FC reverting to hover.
