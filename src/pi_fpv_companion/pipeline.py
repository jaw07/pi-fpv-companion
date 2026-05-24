"""Main loop: camera -> [detector] -> tracker -> visual servo -> safety -> FC backend.

The detector is OPTIONAL because some cameras (IMX500, SyntheticCamera) emit
detections inline in the FrameBundle. When the camera does that, Pipeline
leaves them alone. When it doesn't (PiCam, File, Webcam) and a detector is
configured, Pipeline runs it on the configured cadence.

Detector runs ASYNC by default — inference happens on a worker thread, so the
221 ms NanoDet call on Pi Zero 2W doesn't stall the 30 FPS main loop. Pass
`async_detector=False` for deterministic in-line execution (used by tests).

Generic over camera/detector/tracker/FC implementations. Same Pipeline runs in
dev (SyntheticCamera + FakeArduCopter) and production (PiCamCamera + NanoDet
+ ArduPilotBackend over UART).
"""
from __future__ import annotations
import time
from dataclasses import replace
from typing import Callable, Optional

from pi_fpv_companion.camera.base import Camera, FrameBundle
from pi_fpv_companion.detect.async_detector import AsyncDetector
from pi_fpv_companion.detect.base import Detector
from pi_fpv_companion.guidance.safety import GateResult, SafetyConfig, gate
from pi_fpv_companion.guidance.visual_servo import ServoConfig, compute_intent
from pi_fpv_companion.track.base import Tracker
from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState, Target, ZERO_INTENT


StatusCallback = Callable[
    [Optional[Target], GuidanceIntent, GateResult, SwitchState, bool, FrameBundle], None
]


class Pipeline:
    def __init__(
        self,
        camera: Camera,
        tracker: Tracker,
        servo_cfg: ServoConfig,
        safety_cfg: SafetyConfig,
        fc,
        *,
        detector: Optional[Detector] = None,
        detect_period_frames: int = 1,
        async_detector: bool = True,
        detector_cpu_affinity=None,
        on_status: Optional[StatusCallback] = None,
        force_mode: Optional[GuidanceMode] = None,
        camera_watchdog_s: float = 0.0,
    ) -> None:
        self._force_mode = force_mode
        # Camera stall watchdog (0 = off; production sets ~5s). The IMX500/
        # libcamera frontend can hang with capture_request() blocking the main
        # loop forever — a separate thread is the only way to recover.
        self._camera_watchdog_s = camera_watchdog_s
        self._last_frame_ts = 0.0
        self._got_frame = False
        self._camera = camera
        self._tracker = tracker
        self._servo_cfg = servo_cfg
        self._safety_cfg = safety_cfg
        self._fc = fc
        self._detector = detector
        self._detect_period = max(1, detect_period_frames)
        self._on_status = on_status
        self._stopping = False
        self._frame_idx = -1

        # Alpha-beta filter + wrong-target gating sits between the raw tracker
        # and the servo/safety. Everything downstream consumes FilteredTarget.
        from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
        self._target_filter = AlphaBetaTargetFilter()

        # Async detector worker — runs inference off the main loop thread.
        # Disabled for tests that need deterministic single-threaded semantics.
        self._async_worker: Optional[AsyncDetector] = None
        if detector is not None and async_detector:
            self._async_worker = AsyncDetector(detector, cpu_affinity=detector_cpu_affinity)
            self._async_worker.start()

    def stop(self) -> None:
        self._stopping = True

    def run(self) -> None:
        self._camera.open()
        if self._camera_watchdog_s > 0:
            self._start_camera_watchdog()
        try:
            for bundle in self._camera.frames():
                if self._stopping:
                    break
                self._last_frame_ts = time.monotonic()
                self._got_frame = True
                self.tick(bundle)
        finally:
            if self._async_worker is not None:
                self._async_worker.stop()
            self._camera.close()

    def _start_camera_watchdog(self) -> None:
        """Force a process exit (-> systemd restart) if the camera stalls or
        never delivers a first frame. A stalled IMX500 leaves the main loop
        blocked inside capture_request(), so only a separate thread can act.
        A long startup grace allows the on-sensor firmware upload before frame 1."""
        import os
        import sys
        import threading

        start = time.monotonic()
        grace = max(30.0, self._camera_watchdog_s)

        def _watch() -> None:
            while not self._stopping:
                time.sleep(1.0)
                now = time.monotonic()
                if not self._got_frame:
                    if now - start > grace:
                        print(f"CAMERA WATCHDOG: no first frame within {grace:.0f}s; "
                              "exiting for restart", file=sys.stderr, flush=True)
                        os._exit(2)
                elif now - self._last_frame_ts > self._camera_watchdog_s:
                    print(f"CAMERA WATCHDOG: camera stalled (>{self._camera_watchdog_s:.1f}s "
                          "no frame); exiting for restart", file=sys.stderr, flush=True)
                    os._exit(1)

        threading.Thread(target=_watch, daemon=True, name="camera-watchdog").start()

    def tick(self, bundle: FrameBundle) -> GateResult:
        """One iteration. Exposed so tests can drive the pipeline frame-by-frame."""
        self._frame_idx += 1
        now = bundle.timestamp

        # Use the camera's intrinsic detections if it produced any; otherwise
        # the configured detector runs (async-by-default) on the scheduled cadence.
        detections = list(bundle.detections)
        if not detections and self._detector is not None:
            if self._async_worker is not None:
                # Submit a fresh inference job at the cadence; pick up any
                # completed result this tick (it lands when the worker is done,
                # which is usually a few frames later than the submission).
                if self._frame_idx % self._detect_period == 0:
                    self._async_worker.submit(bundle.image)
                fresh = self._async_worker.poll()
                if fresh is not None:
                    detections = fresh
            else:
                # Synchronous fallback — blocks the loop during inference.
                if self._frame_idx % self._detect_period == 0:
                    detections = self._detector.detect(bundle.image)

        raw_target = self._tracker.consume(bundle.image, detections, now)

        # Filter + quality-assess. Everything downstream uses the FilteredTarget,
        # never the raw tracker output (audit §4/§5).
        target = self._target_filter.update(
            raw_target, bundle.width, bundle.height, now
        )

        switch = self._fc.read_switch()
        if self._force_mode is not None:
            switch = replace(switch, mode=self._force_mode,
                             active=self._force_mode is not GuidanceMode.STANDBY)
        armed = self._fc.is_armed()
        if target is not None:
            # Preview the intent even in STANDBY (using TRACK behaviour) so the
            # HUD shows what guidance would do; the gate decides what's actually
            # sent. When engaged, the switch mode (TRACK/DIVE) drives closure.
            preview_mode = (
                switch.mode if switch.mode is not GuidanceMode.STANDBY else GuidanceMode.TRACK
            )
            intent = compute_intent(target, self._servo_cfg, preview_mode)
        else:
            intent = ZERO_INTENT
        gated = gate(intent, target, switch, armed, now, self._safety_cfg)
        # STANDBY -> release to the pilot's radio. Engaged -> only override if the
        # FC is in the flight mode our control_mode expects (control_ready
        # interlock); otherwise release, so we never push sticks into the wrong
        # mode. (When the gate mutes while engaged, gated.intent is neutral -> hold.)
        ready = getattr(self._fc, "control_ready", None)
        if switch.mode is GuidanceMode.STANDBY or (ready is not None and not ready()):
            self._fc.release()
        else:
            self._fc.send_intent(gated.intent)

        if self._on_status is not None:
            self._on_status(target, intent, gated, switch, armed, bundle)

        return gated
