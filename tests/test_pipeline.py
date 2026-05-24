"""Pipeline integration tests. Covers:
  - IMX500-style path (camera produces detections, no detector argument)
  - PiCam-style path (camera yields raw frames, Pipeline runs detector periodically)
  - KCF re-seeding on detection bursts
  - Safety gate mutes intent correctly
"""
from __future__ import annotations
from typing import List

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.guidance.safety import SafetyConfig
from pi_fpv_companion.guidance.visual_servo import ServoConfig
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.track.kcf_tracker import KcfTracker
from pi_fpv_companion.types import Detection, GuidanceIntent, GuidanceMode, SwitchState


class StubFC:
    def __init__(self, switch_active: bool = True, armed: bool = True) -> None:
        self.switch_active = switch_active
        self.armed = armed
        self.sent: List[GuidanceIntent] = []
        self.released = 0                      # times release() (handback) was called
        self.mode = None                       # if set, overrides switch_active
        self.ready = True                      # control_ready interlock value

    def open(self) -> None: ...
    def close(self) -> None: ...

    def read_switch(self) -> SwitchState:
        mode = self.mode if self.mode is not None else (
            GuidanceMode.TRACK if self.switch_active else GuidanceMode.STANDBY
        )
        return SwitchState(active=mode is not GuidanceMode.STANDBY, pwm_us=1800,
                           timestamp=0.0, mode=mode)

    def is_armed(self) -> bool:
        return self.armed

    def control_ready(self) -> bool:
        return self.ready

    def release(self) -> None:
        self.released += 1

    def send_intent(self, intent: GuidanceIntent) -> None:
        self.sent.append(intent)


class CountingDetector:
    """Detector stub: emits one detection at the configured location, counts calls."""
    def __init__(self, x=200, y=200, w=60, h=60):
        self._x, self._y, self._w, self._h = x, y, w, h
        self.call_count = 0

    def detect(self, image):
        self.call_count += 1
        return [Detection(x=self._x, y=self._y, w=self._w, h=self._h,
                          confidence=0.9, class_id=0, class_name="t")]


def _servo(width=720, height=576):
    return ServoConfig(
        frame_width=width, frame_height=height,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=10.0, yaw_p_gain=0.3, yaw_ff_gain=0.0,
        desired_bbox_frac=0.30, closure_p_gain=50.0,
    )


def _safety():
    return SafetyConfig(watchdog_timeout_s=1.0, require_armed=True)


# ---- IMX500-style (camera produces detections, no detector) ----

def test_imx500_path_locks_and_emits_intent():
    cam = SyntheticCamera(width=720, height=576)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=10)
    fc = StubFC()
    pipeline = Pipeline(cam, tracker, _servo(), _safety(), fc)
    gated = pipeline.tick(cam.render_at(0.0))
    assert not gated.muted


def test_imx500_path_drives_yaw_as_target_drifts():
    cam = SyntheticCamera(width=720, height=576)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=10)
    fc = StubFC()
    pipeline = Pipeline(cam, tracker, _servo(), _safety(), fc)
    for i in range(30):
        pipeline.tick(cam.render_at(i * 0.05))
    assert fc.sent[-1].yaw_rate_dps != 0.0


# ---- PiCam-style (Pipeline runs detector periodically) ----

def test_pipeline_runs_detector_only_on_period_boundary():
    """Pipeline with detect_period_frames=4 should call detector on frames 0, 4, 8, ..."""
    raw_frame = np.full((576, 720, 3), 64, dtype=np.uint8)
    def bundles():
        for i in range(13):
            yield FrameBundle(image=raw_frame, width=720, height=576,
                              timestamp=i * 0.033, detections=[])

    detector = CountingDetector()
    tracker = KcfTracker(max_lost_frames=50)
    fc = StubFC()

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): return bundles()

    pipeline = Pipeline(StubCamera(), tracker, _servo(), _safety(), fc,
                        detector=detector, detect_period_frames=4, async_detector=False)
    for b in bundles():
        pipeline.tick(b)

    # 13 frames at period 4 -> calls on frames 0, 4, 8, 12 = 4 calls
    assert detector.call_count == 4


def test_pipeline_does_not_run_detector_when_camera_provided_detections():
    """When the camera produces detections inline (IMX500 path), Pipeline must NOT
    also run the configured detector — that would waste CPU and double-count."""
    raw_frame = np.full((576, 720, 3), 64, dtype=np.uint8)
    bundle = FrameBundle(
        image=raw_frame, width=720, height=576, timestamp=0.0,
        detections=[Detection(x=300, y=300, w=40, h=40, confidence=0.9, class_id=0, class_name="t")],
    )

    detector = CountingDetector()
    tracker = IouAssociator(iou_threshold=0.2)
    fc = StubFC()

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): yield bundle

    pipeline = Pipeline(StubCamera(), tracker, _servo(), _safety(), fc,
                        detector=detector, detect_period_frames=1, async_detector=False)
    pipeline.tick(bundle)
    assert detector.call_count == 0


def test_pipeline_kcf_path_reseeds_on_detection_burst():
    """KCF locked to one position; on a later frame the detector returns a
    detection at a new position — KCF should re-seed to the new box (refresh scale)."""
    detector = CountingDetector(x=200, y=200, w=60, h=60)
    tracker = KcfTracker(max_lost_frames=50)
    fc = StubFC()

    def make_frame(cx, cy):
        img = np.full((480, 640, 3), 64, dtype=np.uint8)
        img[cy - 30:cy + 30, cx - 30:cx + 30] = (0, 0, 255)
        return img

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): return iter([])

    pipeline = Pipeline(
        StubCamera(), tracker, _servo(width=640, height=480), _safety(), fc,
        detector=detector, detect_period_frames=3, async_detector=False,
    )

    # Frame 0: detector runs at (200,200), tracker locks
    pipeline.tick(FrameBundle(image=make_frame(200, 200), width=640, height=480, timestamp=0.0))
    assert tracker.is_locked()

    # Frame 1, 2: no detector run (period=3, only frame 0,3,6,... fire)
    pipeline.tick(FrameBundle(image=make_frame(205, 200), width=640, height=480, timestamp=0.033))
    pipeline.tick(FrameBundle(image=make_frame(210, 200), width=640, height=480, timestamp=0.066))

    # Frame 3: detector runs again. Reposition the detection.
    detector._x = 250
    detector._y = 210
    pipeline.tick(FrameBundle(image=make_frame(250, 210), width=640, height=480, timestamp=0.099))

    # Detector should have run twice (frames 0 and 3)
    assert detector.call_count == 2


# ---- async detector path ----

def test_pipeline_async_path_eventually_produces_detections():
    """With async_detector=True the worker thread runs inference. After a few
    ticks the tracker should have received at least one detection burst."""
    import time as _time
    raw_frame = np.full((480, 640, 3), 64, dtype=np.uint8)

    class SlowCountingDetector:
        def __init__(self):
            self.call_count = 0
        def detect(self, image):
            _time.sleep(0.005)  # simulate inference cost
            self.call_count += 1
            return [Detection(x=300, y=200, w=40, h=40, confidence=0.9, class_id=0, class_name="t")]

    detector = SlowCountingDetector()
    tracker = IouAssociator(iou_threshold=0.2)
    fc = StubFC()

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): return iter([])

    pipeline = Pipeline(
        StubCamera(), tracker, _servo(width=640, height=480), _safety(), fc,
        detector=detector, detect_period_frames=3, async_detector=True,
    )
    try:
        for i in range(30):
            pipeline.tick(FrameBundle(
                image=raw_frame, width=640, height=480,
                timestamp=i * 0.033, detections=[],
            ))
            _time.sleep(0.005)  # let worker make progress
        assert detector.call_count >= 1
        assert tracker.is_locked()
    finally:
        if pipeline._async_worker is not None:
            pipeline._async_worker.stop()


# ---- safety gate ----

def test_safety_mutes_when_standby():
    cam = SyntheticCamera()
    fc = StubFC(switch_active=False)
    pipeline = Pipeline(cam, IouAssociator(), _servo(), _safety(), fc)
    gated = pipeline.tick(cam.render_at(0.0))
    assert gated.muted
    assert gated.reason == "standby"
    # STANDBY hands control back to the pilot (release), never commands.
    assert fc.released >= 1
    assert fc.sent == []


def test_safety_mutes_when_disarmed():
    cam = SyntheticCamera()
    fc = StubFC(armed=False)
    pipeline = Pipeline(cam, IouAssociator(), _servo(), _safety(), fc)
    gated = pipeline.tick(cam.render_at(0.0))
    assert gated.muted
    assert gated.reason == "fc not armed"


def test_engaged_but_fc_wrong_mode_releases_not_commands():
    # Interlock: engaged (TRACK) but the FC isn't in the expected flight mode ->
    # pipeline releases to the pilot instead of overriding.
    cam = SyntheticCamera()
    fc = StubFC()                 # switch_active=True -> TRACK (engaged)
    fc.ready = False              # control_ready() interlock trips
    pipeline = Pipeline(cam, IouAssociator(), _servo(), _safety(), fc)
    pipeline.tick(cam.render_at(0.0))
    assert fc.released >= 1 and fc.sent == []
    # once the FC is in the right mode, it commands again
    fc.ready = True
    pipeline.tick(cam.render_at(0.05))
    assert len(fc.sent) == 1


def test_standby_releases_engaged_commands():
    cam = SyntheticCamera()
    fc = StubFC()
    fc.mode = GuidanceMode.STANDBY
    pipeline = Pipeline(cam, IouAssociator(), _servo(), _safety(), fc)
    pipeline.tick(cam.render_at(0.00))          # STANDBY -> release, no command
    assert fc.released == 1 and fc.sent == []
    fc.mode = GuidanceMode.TRACK
    pipeline.tick(cam.render_at(0.05))          # engaged -> command sent
    assert fc.released == 1 and len(fc.sent) == 1
    fc.mode = GuidanceMode.DIVE
    pipeline.tick(cam.render_at(0.10))          # still engaged -> another command
    assert fc.released == 1 and len(fc.sent) == 2
    fc.mode = GuidanceMode.STANDBY
    pipeline.tick(cam.render_at(0.15))          # back to STANDBY -> release again
    assert fc.released == 2 and len(fc.sent) == 2
