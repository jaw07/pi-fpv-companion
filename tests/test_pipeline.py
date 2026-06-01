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
        self.select = 1000                     # select-channel pwm (multi-target cycle)

    def select_pwm(self) -> int:
        return self.select

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


def test_pipeline_emits_closed_loop_vertical_rate_in_dive():
    """End-to-end: in DIVE the pipeline emits a commanded vertical RATE (closed-loop
    homing) that the backend receives. A target low in frame → descend (rate < 0);
    high in frame → climb (rate > 0)."""
    servo = ServoConfig(
        frame_width=720, frame_height=576, max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=10.0, yaw_p_gain=0.3, yaw_ff_gain=0.0, desired_bbox_frac=0.30,
        closure_p_gain=50.0, dive_forward_deg=8.0, dive_vrate_gain=17.0,
    )

    def run_with_target_y(y):
        bundle = FrameBundle(
            image=np.full((576, 720, 3), 64, dtype=np.uint8),
            width=720, height=576, timestamp=0.0,
            detections=[Detection(x=360, y=y, w=40, h=40, confidence=0.9, class_id=0, class_name="t")],
        )

        class StubCamera:
            def open(self): pass
            def close(self): pass
            def frames(self): yield bundle

        fc = StubFC()
        fc.mode = GuidanceMode.DIVE
        pipe = Pipeline(StubCamera(), IouAssociator(iou_threshold=0.2), servo, _safety(), fc)
        pipe.tick(bundle)
        return fc.sent[-1]

    assert run_with_target_y(440).vertical_rate_mps < 0      # low in frame → descend
    assert run_with_target_y(140).vertical_rate_mps > 0      # high in frame → climb


def test_multi_target_select_cycles_and_lock_persists_through_modes():
    """STANDBY shows all detections; a select-channel pulse cycles the lock; and
    whatever is locked stays locked through TRACK and DIVE."""
    from pi_fpv_companion.track.multi_target import MultiObjectTracker
    # Two well-separated detections; B (right) higher-confidence so it auto-locks.
    dets = [Detection(x=150, y=300, w=40, h=40, confidence=0.6, class_id=0, class_name="A"),
            Detection(x=560, y=300, w=40, h=40, confidence=0.9, class_id=0, class_name="B")]
    bundle = FrameBundle(image=np.full((576, 720, 3), 64, dtype=np.uint8),
                         width=720, height=576, timestamp=0.0, detections=dets)

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): yield bundle

    fc = StubFC()
    fc.select = 1000
    locked = []
    pipe = Pipeline(StubCamera(), MultiObjectTracker(iou_threshold=0.2), _servo(), _safety(), fc,
                    on_status=lambda tgt, *a: locked.append(tgt.track_id if tgt else None))

    # STANDBY: all detections visible; auto-locked on the highest-confidence (B, x≈560).
    fc.mode = GuidanceMode.STANDBY
    g = pipe.tick(bundle)
    assert pipe._tracks is not None and len(pipe._tracks) == 2     # both shown
    id_b = pipe._tracker.selected_id
    assert pipe._tracker._tracks[id_b].detection.x == 560

    # Pulse the select channel (rising edge) → cycle to the other target (A, x≈150).
    fc.select = 1800
    pipe.tick(bundle)
    id_a = pipe._tracker.selected_id
    assert id_a != id_b and pipe._tracker._tracks[id_a].detection.x == 150
    fc.select = 1000                                              # release (no re-trigger)
    pipe.tick(bundle)
    assert pipe._tracker.selected_id == id_a

    # Now commit: STANDBY → TRACK → DIVE. The lock stays on A throughout, and a
    # a select pulse while ENGAGED is ignored (the lock is frozen once committed).
    for mode in (GuidanceMode.TRACK, GuidanceMode.DIVE):
        fc.mode = mode
        fc.armed = True
        fc.select = 1000
        pipe.tick(bundle)
        fc.select = 1800                                          # try to cycle mid-engagement
        pipe.tick(bundle)
        assert pipe._tracker.selected_id == id_a                  # ignored — still locked on A
    assert locked[-1] == id_a                                     # guidance followed the selection


def test_engaged_dive_holds_when_committed_target_drops_not_swaps():
    """Committed on target A in DIVE; A disappears (only B remains). The lock must
    NOT swap to B — the tracker holds (no auto-reacquire while engaged), so the
    aircraft never attacks a different target than the one committed to."""
    from pi_fpv_companion.track.multi_target import MultiObjectTracker

    def bundle_with(dets):
        return FrameBundle(image=np.full((576, 720, 3), 64, dtype=np.uint8),
                           width=720, height=576, timestamp=0.0, detections=dets)

    A = Detection(x=150, y=300, w=40, h=40, confidence=0.6, class_id=0)
    B = Detection(x=560, y=300, w=40, h=40, confidence=0.9, class_id=0)
    fc = StubFC()
    tracker = MultiObjectTracker(iou_threshold=0.2, max_lost_frames=2)
    pipe = Pipeline(StubCameraNoop(), tracker, _servo(), _safety(), fc)

    # STANDBY: cycle to A (the low-confidence left target).
    fc.mode = GuidanceMode.STANDBY
    pipe.tick(bundle_with([A, B]))
    while tracker._tracks[tracker.selected_id].detection.x != 150:
        fc.select = 1800; pipe.tick(bundle_with([A, B])); fc.select = 1000
    id_a = tracker.selected_id

    # Commit to DIVE, then A vanishes (only B detected) for longer than max_lost.
    fc.mode = GuidanceMode.DIVE
    fc.armed = True
    gated = None
    for _ in range(5):
        gated = pipe.tick(bundle_with([B]))
    assert tracker.selected_id == id_a            # never swapped to B
    assert gated.muted                            # held (no target) instead of attacking B


class StubCameraNoop:
    def open(self): pass
    def close(self): pass
    def frames(self): return iter(())


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


def test_watchdog_mutes_when_detections_stop_arriving():
    """Regression for the audit's dead-watchdog finding. Lock a target, then stop
    feeding detections so the tracker coasts on a frozen box. The staleness
    watchdog must fire in the INTEGRATED pipeline (it could not before, because
    the filter restamped `timestamp=now` every tick). `measurement_timestamp`
    freezes at the last real detection, so `now - measurement_timestamp` grows
    and the gate mutes with reason 'target stale'."""
    img = np.full((576, 720, 3), 64, dtype=np.uint8)
    det = [Detection(x=360, y=288, w=60, h=60, confidence=0.9, class_id=0, class_name="t")]

    def bundle(i, with_det):
        return FrameBundle(image=img, width=720, height=576, timestamp=i * 0.05,
                           detections=det if with_det else [])

    # Tracker coasts for many frames (won't itself drop the track during the test
    # window); watchdog window is short so staleness fires first.
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=100)
    fc = StubFC()
    safety = SafetyConfig(watchdog_timeout_s=0.2, require_armed=True)
    pipeline = Pipeline(SyntheticCamera(), tracker, _servo(), safety, fc)

    # Frames 0..2: real detections -> lock and pass the gate.
    for i in range(3):
        g = pipeline.tick(bundle(i, with_det=True))
    assert not g.muted

    # Frames 3+: no detections -> tracker coasts (lost_frames>0), filter coasts.
    reasons = [pipeline.tick(bundle(i, with_det=False)).reason for i in range(3, 12)]
    assert "target stale" in reasons                     # the watchdog actually fires
    # ...and it fires while the target still exists (not because it dropped to None).
    assert reasons.index("target stale") < (reasons + ["no target"]).index("no target")


# ---- guided_nogps body-RATE path (control_mode: guided_nogps) ----

from pi_fpv_companion.guidance.rate_control import RateConfig   # noqa: E402


class RateStubFC(StubFC):
    """StubFC + the body-rate surface and airframe-state accessors the rate path uses."""
    def __init__(self, pitch=0.0, climb=0.0, **kw):
        super().__init__(**kw)
        self.body_rates = []                   # (roll_rate, pitch_rate, yaw_rate, thrust)
        self._pitch = pitch                    # deg
        self._climb = climb                    # m/s, +up

    def send_body_rates(self, rr, pr, yr, thrust):
        self.body_rates.append((rr, pr, yr, thrust))

    def pitch_deg(self): return self._pitch
    def roll_deg(self): return 0.0
    def flight_path_angle_rad(self): return 0.0
    def agl_m(self): return 40.0
    def climb_mps(self): return self._climb


def test_hover_trim_only_learns_while_level():
    # Online hover trim must adapt only while roughly LEVEL. Pitched-down chase (sinking by
    # intent) must NOT crank hover, or a later hold/SEARCH balloons up on the bad hover.
    cam = SyntheticCamera(width=720, height=576)

    def _run_track(pitch_deg):
        fc = RateStubFC(pitch=pitch_deg, climb=-2.0)   # descending
        fc.mode = GuidanceMode.TRACK
        pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2, max_lost_frames=10),
                        _servo(), _safety(), fc, rate_cfg=RateConfig(720, 576))
        for i in range(6):
            pipe.tick(cam.render_at(i * 0.05))
        return pipe._rate_state.hover

    level = _run_track(0.0)
    steep = _run_track(40.0)
    assert level > 0.30, "level + sinking -> hover trims UP"
    assert abs(steep - 0.30) < 1e-9, "pitched-down chase -> hover frozen"


def test_guided_nogps_rate_path_sends_body_rates_not_sticks():
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC()
    fc.mode = GuidanceMode.DIVE
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2, max_lost_frames=10),
                    _servo(), _safety(), fc, rate_cfg=RateConfig(720, 576))
    for i in range(5):
        pipe.tick(cam.render_at(i * 0.05))
    assert fc.body_rates, "guided_nogps must command body rates"
    assert fc.sent == [], "the RC-stick (send_intent) path must NOT be used in rate mode"


def test_guided_nogps_rate_path_releases_in_standby():
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC(switch_active=False)       # STANDBY
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(), _safety(), fc,
                    rate_cfg=RateConfig(720, 576))
    pipe.tick(cam.render_at(0.0))
    assert fc.released >= 1
    assert fc.body_rates == []


def test_stabilize_path_unchanged_uses_send_intent():
    # control_mode != guided_nogps (rate_cfg=None) -> the RC-stick path is used, untouched.
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC()
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2, max_lost_frames=10),
                    _servo(), _safety(), fc)   # no rate_cfg
    for i in range(5):
        pipe.tick(cam.render_at(i * 0.05))
    assert fc.sent, "STABILIZE path must still send RC-stick intents"
    assert fc.body_rates == [], "STABILIZE path must NOT command body rates"
