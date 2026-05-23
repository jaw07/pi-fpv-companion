"""Standard Raspberry Pi Camera via picamera2.

PiCam path: yields raw frames; Pipeline runs the configured detector on the
periodic cadence. For the IMX500 sensor, use `IMX500Camera` instead.

Three things this module gets right that are easy to get wrong:

1. **Sensor mode selection for FOV.** libcamera's default sensor mode for a
   small output is often a *cropped* readout (e.g. IMX708's 1536x864 mode only
   covers 3072x1728 of the 4608x2592 array — a ~67% centre crop, looks
   "zoomed in"). We pick the lowest-resolution mode whose `crop_limits` cover
   the FULL pixel array, so we get the widest FOV the lens allows, then
   downscale to the output size. `ScalerCrop` can't fix this — it only selects
   *within* the active sensor mode's window.

2. **Control lifecycle.** picamera2 runtime controls (AeExposureMode,
   NoiseReductionMode, ...) set via `set_controls()` *between* `configure()`
   and `start()` are silently ignored. They must go in the configuration's
   `controls=` dict. (Verified on hardware — `set_controls()` pre-start was a
   no-op.)

3. **Latest-frame capture.** A background thread drains libcamera flat-out
   into a single-slot buffer; `frames()` always yields the freshest frame and
   drops what the pipeline couldn't keep up with. Stale frames = stale
   guidance, so dropping (not queueing) is correct flight behavior. The thread
   surfaces capture faults loud instead of silently stalling the aircraft.

`picamera2` is Pi-only (apt `python3-picamera2`, not pip). Imports are lazy.
"""
from __future__ import annotations
import threading
import time
from typing import Iterator, Optional

from pi_fpv_companion.camera.base import FrameBundle

_EXPOSURE_MODES = {"normal": 0, "short": 1, "long": 2}
_NOISE_MODES = {"off": 0, "fast": 1, "high_quality": 2}
_STALE_FRAME_TIMEOUT_S = 2.0   # no new frame for this long => surface a fault


class PiCamCamera:
    def __init__(
        self,
        width: int = 720,
        height: int = 576,
        framerate: int = 30,
        exposure_mode: str = "short",
        noise_reduction: str = "fast",
        hflip: bool = False,
        vflip: bool = False,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = framerate
        self._exposure_mode = exposure_mode
        self._noise_reduction = noise_reduction
        self._hflip = hflip
        self._vflip = vflip
        self._picam = None
        self._running = False
        self._latest = None
        self._latest_seq = 0
        self._last_frame_mono = 0.0
        self._capture_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._cap_stop = threading.Event()
        self._cap_thread: Optional[threading.Thread] = None

    def _pick_full_fov_mode(self, picam):
        """Lowest-res sensor mode whose crop covers the full array (widest FOV).
        Returns the mode's size tuple, or None to let libcamera choose.

        Takes `picam` explicitly — during open() self._picam isn't assigned
        until the end, so referencing self._picam here would always be None.
        """
        try:
            pa_w, pa_h = picam.camera_properties["PixelArraySize"]
            modes = picam.sensor_modes
        except (KeyError, AttributeError):
            return None
        full = []
        for m in modes:
            cl = m.get("crop_limits")
            sz = m.get("size")
            if not cl or not sz:
                continue
            if cl[2] >= pa_w * 0.98 and cl[3] >= pa_h * 0.98:
                full.append(sz)
        if not full:
            return None
        return min(full, key=lambda s: s[0] * s[1])

    def open(self) -> None:
        if self._picam is not None:
            raise RuntimeError("PiCamCamera.open() called twice without close()")
        from picamera2 import Picamera2  # lazy
        from libcamera import Transform   # lazy

        picam = Picamera2()
        try:
            controls = {"FrameRate": float(self._fps)}
            em = _EXPOSURE_MODES.get(self._exposure_mode)
            if em is None:
                print(f"WARN: unknown camera.exposure_mode "
                      f"{self._exposure_mode!r}; using sensor default")
            else:
                controls["AeExposureMode"] = em
            nr = _NOISE_MODES.get(self._noise_reduction)
            if nr is None:
                print(f"WARN: unknown camera.noise_reduction "
                      f"{self._noise_reduction!r}; using sensor default")
            else:
                controls["NoiseReductionMode"] = nr

            raw_size = self._pick_full_fov_mode(picam)
            kwargs = dict(
                main={"size": (self._width, self._height), "format": "BGR888"},
                controls=controls,
                transform=Transform(hflip=self._hflip, vflip=self._vflip),
                buffer_count=4,
            )
            if raw_size is not None:
                kwargs["raw"] = {"size": raw_size}

            config = picam.create_preview_configuration(**kwargs)
            picam.configure(config)
            picam.start()
        except Exception:
            try:
                picam.close()
            except Exception:
                pass
            raise

        self._picam = picam
        self._running = True
        self._last_frame_mono = time.monotonic()
        self._cap_stop.clear()
        self._cap_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="picam-capture"
        )
        self._cap_thread.start()

    def _capture_loop(self) -> None:
        consecutive_errors = 0
        while not self._cap_stop.is_set():
            try:
                arr = self._picam.capture_array("main")
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    with self._lock:
                        self._capture_error = e
                    return
                time.sleep(0.05)   # back off; don't peg a core on fast failures
                continue
            consecutive_errors = 0
            with self._lock:
                self._latest = arr
                self._latest_seq += 1
                self._last_frame_mono = time.monotonic()

    def close(self) -> None:
        self._running = False
        self._cap_stop.set()
        alive = False
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=3.0)
            alive = self._cap_thread.is_alive()
            self._cap_thread = None
        if self._picam is not None:
            if alive:
                # Capture thread is wedged inside libcamera; tearing the camera
                # down underneath it risks a use-after-free. Leak the handle
                # rather than crash — process is shutting down anyway.
                print("WARN: picam capture thread did not exit; "
                      "skipping camera teardown to avoid libcamera UAF")
            else:
                try:
                    self._picam.stop()
                finally:
                    self._picam.close()
            self._picam = None

    def frames(self) -> Iterator[FrameBundle]:
        if not self._running:
            self.open()
        last_seq = -1
        while self._running:
            with self._lock:
                arr = self._latest
                seq = self._latest_seq
                err = self._capture_error
                since = time.monotonic() - self._last_frame_mono
            if err is not None:
                raise RuntimeError(
                    "PiCam capture thread failed (libcamera error)"
                ) from err
            if arr is None or seq == last_seq:
                # Camera-dead is correctly fatal (no video = unrecoverable
                # in-process; let systemd restart + re-init the camera).
                # `_last_frame_mono` is set at open(), so this also catches
                # "started but never delivered a first frame", not just
                # "delivered frames then stalled".
                if since > _STALE_FRAME_TIMEOUT_S:
                    what = "stalled" if arr is not None else "never delivered a frame"
                    raise RuntimeError(
                        f"PiCam {what} ({since:.1f}s, libcamera fault)"
                    )
                time.sleep(0.003)
                continue
            last_seq = seq
            h, w = arr.shape[:2]
            yield FrameBundle(
                image=arr, width=w, height=h,
                timestamp=time.monotonic(),
                detections=[],
            )
