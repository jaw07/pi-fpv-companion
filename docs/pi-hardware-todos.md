# Things to validate on real Pi Zero 2W hardware

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
