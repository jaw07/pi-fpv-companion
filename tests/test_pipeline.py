"""Pipeline integration tests. Covers:
  - IMX500-style path (camera produces detections, no detector argument)
  - dev-camera path (file/webcam yields raw frames, Pipeline runs detector periodically)
  - KCF re-seeding on detection bursts
  - Safety gate mutes intent correctly
"""
from __future__ import annotations
import time
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
        self.engaged_calls: List[bool] = []    # set_engaged(bool) history (auto-engage edge)

    def select_pwm(self) -> int:
        return self.select

    def set_engaged(self, engaged: bool) -> None:
        self.engaged_calls.append(engaged)

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


# ---- dev-camera path: Pipeline runs the detector inline (file/webcam) ----

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
                        detector=detector, detect_period_frames=4)
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
                        detector=detector, detect_period_frames=1)
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


def test_auto_engage_fires_set_engaged_only_on_standby_engaged_edges():
    """ch7 auto-engage: set_engaged(True) on STANDBY->engaged, set_engaged(False) on
    the way back — once per transition, never every frame (TRACK<->DIVE doesn't re-fire)."""
    from pi_fpv_companion.track.multi_target import MultiObjectTracker
    dets = [Detection(x=360, y=300, w=40, h=40, confidence=0.9, class_id=0)]
    bundle = FrameBundle(image=np.full((576, 720, 3), 64, dtype=np.uint8),
                         width=720, height=576, timestamp=0.0, detections=dets)

    class StubCamera:
        def open(self): pass
        def close(self): pass
        def frames(self): yield bundle

    fc = StubFC()
    pipe = Pipeline(StubCamera(), MultiObjectTracker(iou_threshold=0.2), _servo(), _safety(), fc)
    for mode in (GuidanceMode.STANDBY,            # no edge (starts disengaged)
                 GuidanceMode.TRACK,              # -> engaged  => set_engaged(True)
                 GuidanceMode.DIVE,               # still engaged => no call
                 GuidanceMode.STANDBY):           # -> disengaged => set_engaged(False)
        fc.mode = mode
        pipe.tick(bundle)
    assert fc.engaged_calls == [True, False]


def test_engaged_dive_holds_when_committed_target_drops_not_swaps():
    """Committed on target A in DIVE; A disappears (only B remains). The lock must
    NOT swap to B — the tracker holds (no auto-reacquire while engaged), so the
    aircraft never attacks a different target than the one committed to."""
    from pi_fpv_companion.track.multi_target import MultiObjectTracker

    def bundle_with(dets, t=0.0):
        return FrameBundle(image=np.full((576, 720, 3), 64, dtype=np.uint8),
                           width=720, height=576, timestamp=t, detections=dets)

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
    # Time ADVANCES (real frames carry a clock) so the lost target ages out via the
    # staleness watchdog / time-based quality — the realistic mute mechanism.
    fc.mode = GuidanceMode.DIVE
    fc.armed = True
    gated = None
    for i in range(5):
        gated = pipe.tick(bundle_with([B], t=0.5 * (i + 1)))
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
        detector=detector, detect_period_frames=3,
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


def test_guided_nogps_standby_injects_nothing_even_in_guided():
    # Operator requirement (flight-2 hardening): STANDBY injects NOTHING, regardless
    # of the FC's flight mode. Even armed with the FC left in GUIDED_NOGPS, the
    # companion is silent — ArduCopter's GUID_TIMEOUT hold (auto-enforced to 3 s)
    # catches the craft after the last engaged setpoint.
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC(switch_active=False, pitch=10.0)   # STANDBY, armed, FC in GUIDED_NOGPS
    fc.ready = True
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(), _safety(), fc,
                    rate_cfg=RateConfig(720, 576))
    for t in (0.0, 0.5, 3.0, 30.0):
        pipe.tick(cam.render_at(t))
    assert fc.body_rates == [], "STANDBY must NEVER inject commands, in any FC mode"
    assert fc.released >= 4                            # hands-off is the only transmission


def test_guided_nogps_standby_hover_hold_never_sent_while_disarmed():
    # Flight-2 fix: a standing hover-thrust SET_ATTITUDE_TARGET while DISARMED on the
    # ground (FC left in GUIDED_NOGPS) means the craft launches itself the instant the
    # pilot arms. STANDBY + in-guided + disarmed -> release only, NO body rates.
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC(switch_active=False, armed=False)
    fc.ready = True                                    # FC IS in GUIDED_NOGPS
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(), _safety(), fc,
                    rate_cfg=RateConfig(720, 576))
    pipe.tick(cam.render_at(0.0))
    assert fc.released >= 1
    assert fc.body_rates == [], "disarmed must NEVER receive a thrust setpoint"


def test_guided_nogps_engaged_sends_nothing_while_disarmed():
    # Same guard on the engaged path: the muted SAFE HOLD must not transmit hover
    # thrust to a disarmed FC either (bench: engaged + props off + disarmed).
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC(armed=False)
    fc.mode = GuidanceMode.TRACK
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2, max_lost_frames=10),
                    _servo(), _safety(), fc, rate_cfg=RateConfig(720, 576))
    for i in range(3):
        pipe.tick(cam.render_at(i * 0.05))
    assert fc.body_rates == [], "disarmed must NEVER receive body-rate commands"
    # And the controller state stays pristine: no integrals/timers wound while
    # disarmed that would release as a step input on the first armed tick.
    assert pipe._rate_state.last_t is None and pipe._rate_state.engage_h is None
    assert abs(pipe._rate_state.hover - 0.30) < 1e-9


def test_stabilize_path_releases_instead_of_hover_override_while_disarmed():
    # ZERO_INTENT's thrust is HOVER (0.5): in stabilize that maps to the hover
    # throttle PWM, so sending the muted intent to a disarmed FC is a standing
    # throttle-at-hover override -> self-launch at arm. Engaged + disarmed must
    # RELEASE the channels, not send the neutral intent.
    cam = SyntheticCamera(width=720, height=576)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=10)
    fc = StubFC(armed=False)               # engaged (TRACK), control_ready, DISARMED
    pipeline = Pipeline(cam, tracker, _servo(), _safety(), fc)
    for i in range(3):
        pipeline.tick(cam.render_at(i * 0.05))
    assert fc.sent == [], "disarmed must NEVER receive stick overrides"
    assert fc.released >= 1


def test_guided_nogps_standby_commands_nothing_when_pilot_takes_manual():
    # FC NOT in GUIDED_NOGPS (pilot flipped the mode away = manual recovery) -> command NOTHING.
    cam = SyntheticCamera(width=720, height=576)
    fc = RateStubFC(switch_active=False)
    fc.ready = False                                   # control_ready() False = FC not in guided
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


# --------------------- decoupled capture/display + control split ---------------------

def test_decoupled_run_renders_at_capture_rate_and_controls_separately():
    """With `display` set, run() splits capture+display from the control tick onto
    separate threads. Verify: both run, the camera is opened/closed, and the FC is
    driven ONLY by the control thread (FC interactions == control ticks, NOT render
    calls) so the safety contract is unaffected by the faster display path."""
    import time

    class _ListCamera:
        def __init__(self, bundles, delay=0.01):
            self._bundles, self._delay = bundles, delay
            self.opened = self.closed = False
        def open(self): self.opened = True
        def close(self): self.closed = True
        def frames(self):
            for b in self._bundles:
                time.sleep(self._delay)
                yield b

    src = SyntheticCamera(width=320, height=240)
    bundles = [src.render_at(i * 0.05) for i in range(15)]
    cam = _ListCamera(bundles, delay=0.01)
    fc = StubFC(switch_active=True, armed=True)

    displayed, controlled = [], []
    def display(target, intent, gated, switch, armed, frame, tracks=None):
        displayed.append(frame)
    def on_status(target, intent, gated, switch, armed, frame, tracks=None):
        controlled.append(switch.mode)

    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(320, 240), _safety(), fc,
                    on_status=on_status, display=display)
    pipe.run()

    assert cam.opened and cam.closed
    assert len(controlled) >= 1, "control thread must have ticked"
    assert len(displayed) >= 1, "capture thread must have rendered"
    # The FC is touched once per control tick (send_intent or release), never from the
    # render path: total FC interactions equal control ticks, independent of render count.
    assert len(fc.sent) + fc.released == len(controlled)


def test_decoupled_run_stops_cleanly_on_stop():
    """stop() must end the capture loop and join the control thread (no hang/leak)."""
    import time, threading

    class _SlowCamera:
        def __init__(self): self.opened = self.closed = False
        def open(self): self.opened = True
        def close(self): self.closed = True
        def frames(self):
            src = SyntheticCamera(width=320, height=240)
            i = 0
            while True:
                time.sleep(0.01)
                yield src.render_at(i * 0.05); i += 1

    cam = _SlowCamera()
    fc = StubFC(switch_active=False, armed=False)
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(320, 240), _safety(), fc,
                    on_status=lambda *a, **k: None, display=lambda *a, **k: None)
    t = threading.Thread(target=pipe.run)
    t.start()
    time.sleep(0.2)
    pipe.stop()
    t.join(timeout=3.0)
    assert not t.is_alive(), "run() must return promptly after stop()"
    assert cam.closed


# --------------------- event-driven (detection-driven) control loop ---------------------

def test_detection_sig_stable_and_discriminating():
    from pi_fpv_companion.pipeline import Pipeline as _P
    d1 = [Detection(x=100.2, y=200.4, w=40, h=40, confidence=0.9, class_id=0)]
    d1b = [Detection(x=100.1, y=200.3, w=40, h=40, confidence=0.7, class_id=0)]  # ~same box
    d2 = [Detection(x=180, y=200, w=40, h=40, confidence=0.9, class_id=0)]        # moved
    assert _P._detection_sig(d1) == _P._detection_sig(d1b)   # rounds to same box -> same sig
    assert _P._detection_sig(d1) != _P._detection_sig(d2)    # moved target -> fresh
    assert _P._detection_sig([]) == ()                        # empty scene


def _det_bundle(x):
    return FrameBundle(
        image=np.zeros((240, 320, 3), dtype=np.uint8), width=320, height=240,
        timestamp=0.0,
        detections=[Detection(x=x, y=120, w=30, h=30, confidence=0.9, class_id=0, class_name="t")])


def _empty_bundle():
    return FrameBundle(
        image=np.zeros((240, 320, 3), dtype=np.uint8), width=320, height=240,
        timestamp=0.0, detections=[])


class _ListCam:
    def __init__(self, bundles, delay): self._b, self._d = bundles, delay; self.opened=self.closed=False
    def open(self): self.opened=True
    def close(self): self.closed=True
    def frames(self):
        import time as _t
        for b in self._b:
            _t.sleep(self._d); yield b


def test_repeated_detections_still_tick_but_do_not_refeed_the_tracker():
    # A camera whose detector is slower than its frame rate repeats the previous tensor
    # on the frames in between. Two separate requirements, which used to be conflated:
    #
    #   1. the control tick MUST still run every frame — it is what reads the RC switch,
    #      the armed state and drives the FC. Gating it on the detection changing made
    #      the whole loop run at ~5Hz whenever the target was near-stationary, which is
    #      what made the tracker feel laggy and jittery in STANDBY (measured on the
    #      airframe: 0.4px of detector dither -> 5.1Hz control, box frozen 69% of the time).
    #   2. the tracker/filter must NOT be re-fed the repeated tensor — that reads as a
    #      fresh confirmation and would reset the staleness clock the safety gate uses,
    #      so a frozen detector would look healthy.
    bundles = [_det_bundle(160.0) for _ in range(15)]      # identical detection every frame
    cam = _ListCam(bundles, delay=0.02)                    # ~0.3s of frames
    fc = StubFC(switch_active=True, armed=True)
    ticks = []
    tracker = IouAssociator(iou_threshold=0.2)
    consumed = []
    orig_consume = tracker.consume

    def counting_consume(image, detections, now):
        consumed.append(now)
        return orig_consume(image, detections, now)
    tracker.consume = counting_consume

    pipe = Pipeline(cam, tracker, _servo(320, 240), _safety(), fc,
                    on_status=lambda *a, **k: ticks.append(1),
                    display=lambda *a, **k: None)
    pipe._control_fallback_s = 0.1
    pipe.run()

    # 1. the tick ran for essentially every frame, not at the old ~5Hz fallback rate.
    assert len(ticks) >= 12, f"control tick should run per frame, got {len(ticks)}/15"
    # 2. but the repeated tensor was only ever handed to the tracker once.
    assert len(consumed) == 1, f"tracker re-fed a repeated tensor {len(consumed)} times"


def test_empty_detections_keep_feeding_the_tracker():
    # "Nothing detected" repeats the same (empty) signature every frame, but it is real
    # information: without it a coasting track never ages out and the HUD keeps drawing
    # a ghost box over an empty scene forever.
    bundles = [_empty_bundle() for _ in range(6)]
    cam = _ListCam(bundles, delay=0.02)
    fc = StubFC(switch_active=True, armed=True)
    tracker = IouAssociator(iou_threshold=0.2)
    consumed = []
    orig_consume = tracker.consume

    def counting_consume(image, detections, now):
        consumed.append(now)
        return orig_consume(image, detections, now)
    tracker.consume = counting_consume

    pipe = Pipeline(cam, tracker, _servo(320, 240), _safety(), fc,
                    display=lambda *a, **k: None)
    pipe._control_fallback_s = 0.1
    pipe.run()
    assert len(consumed) >= 5, f"empty frames must keep aging the tracker, got {len(consumed)}"


def test_control_loop_ticks_on_each_fresh_detection():
    # Distinct detections (moving target) each tick guidance; fallback set high so the
    # keepalive can't account for the ticks.
    bundles = [_det_bundle(80.0 + 12 * i) for i in range(8)]   # target moves each frame
    cam = _ListCam(bundles, delay=0.05)                        # slow enough the loop keeps up
    fc = StubFC(switch_active=True, armed=True)
    ticks = []
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(320, 240), _safety(), fc,
                    on_status=lambda *a, **k: ticks.append(1),
                    display=lambda *a, **k: None)
    pipe._control_fallback_s = 1.0                             # fallback won't fire in ~0.4s
    pipe.run()
    assert len(ticks) >= 6, f"expected ~one tick per fresh detection, got {len(ticks)}"


# ---- display-side forward prediction -------------------------------------------------

def _ftarget(x, y, vx, vy, ts):
    from pi_fpv_companion.types import FilteredTarget
    return FilteredTarget(
        detection=Detection(x=x, y=y, w=20, h=20, confidence=0.9, class_id=0, class_name="t"),
        track_id=1, vx_px_s=vx, vy_px_s=vy, quality=0.9,
        timestamp=ts, measurement_timestamp=ts)


def test_predict_to_advances_the_drawn_box_to_frame_time():
    # The control thread produces state from frame N while the capture thread renders
    # N+1, so the box trails the live image by one frame. Stepping it along the filter's
    # velocity estimate removes that (airframe A/B: 9.1px behind -> 0.1px at 200px/s).
    from pi_fpv_companion.pipeline import _predict_to
    t = _ftarget(100.0, 50.0, vx=200.0, vy=-40.0, ts=10.0)
    out = _predict_to(t, 10.045)                       # one 45ms frame later
    assert abs(out.detection.x - 109.0) < 0.1
    assert abs(out.detection.y - 48.2) < 0.1
    # everything except the centroid is untouched
    assert out.track_id == t.track_id and out.quality == t.quality
    assert out.measurement_timestamp == t.measurement_timestamp
    assert out.detection.w == t.detection.w and out.detection.h == t.detection.h


def test_predict_to_refuses_to_extrapolate_stale_state():
    # Beyond the horizon the control state is old enough that extrapolating on a stale
    # velocity would fling the box across the screen; freeze it instead.
    from pi_fpv_companion.pipeline import _predict_to, _PREDICT_HORIZON_S
    t = _ftarget(100.0, 50.0, vx=800.0, vy=0.0, ts=10.0)
    out = _predict_to(t, 10.0 + _PREDICT_HORIZON_S + 0.01)
    assert out.detection.x == 100.0                    # unchanged
    assert _predict_to(t, 9.9).detection.x == 100.0    # negative dt -> unchanged
    assert _predict_to(None, 10.0) is None


def test_forward_prediction_is_display_only():
    # Guidance and the safety gate must see the UNMODIFIED target — the prediction is
    # cosmetic and must never influence what is sent to the FC.
    fc = StubFC(switch_active=True, armed=True)
    from_status, from_display = [], []
    cam = _ListCam([_det_bundle(100.0 + 30 * i) for i in range(6)], delay=0.02)
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2), _servo(320, 240), _safety(), fc,
                    on_status=lambda tgt, *a, **k: from_status.append(tgt),
                    display=lambda tgt, *a, **k: from_display.append(tgt))
    pipe.run()
    # The control path sees targets whose centroid is exactly the filter's output: its
    # timestamp equals the bundle timestamp it was computed from (all bundles here are
    # stamped 0.0), so no prediction has been applied.
    live = [t for t in from_status if t is not None]
    assert live, "expected the pipeline to lock a target"
    for t in live:
        assert t.timestamp == 0.0          # untouched by _predict_to
    assert from_display, "expected the display path to be driven"


# ---- camera-stall clock ---------------------------------------------------------------

def test_tick_defaults_its_clock_to_the_frame_timestamp():
    fc = StubFC(switch_active=True, armed=True)
    pipe = Pipeline(_ListCam([], delay=0), IouAssociator(iou_threshold=0.2),
                    _servo(320, 240), _safety(), fc)
    b = _det_bundle(160.0)
    pipe.tick(b)
    # the filter was fed the bundle's own timestamp
    assert pipe._last_target is not None
    assert pipe._last_target.measurement_timestamp == b.timestamp


def test_stalled_camera_still_ages_the_target_into_a_mute():
    """A camera stall re-ticks the LAST bundle. If tick() took its clock from that
    frozen frame, `now` would never advance and the safety gate could not conclude the
    target had gone stale — guidance would keep commanding on a dead image until the
    camera watchdog fired. The control loop therefore advances the clock by real
    elapsed time on a re-tick; this asserts the gate actually mutes because of it."""
    fc = StubFC(switch_active=True, armed=True)
    safety = SafetyConfig(watchdog_timeout_s=0.25, require_armed=True)
    pipe = Pipeline(_ListCam([], delay=0), IouAssociator(iou_threshold=0.2),
                    _servo(320, 240), safety, fc)
    bundle = _det_bundle(160.0)

    fresh = pipe.tick(bundle)                     # frame arrives, target locked
    assert not fresh.muted, f"expected a live target, got {fresh.reason!r}"

    # Frozen frame, clock NOT advanced (the old behaviour) -> gate still thinks it is live.
    assert not pipe.tick(bundle).muted

    # Frozen frame, clock advanced past the watchdog -> must mute as stale.
    stalled = pipe.tick(bundle, bundle.timestamp + 0.3)
    assert stalled.muted and stalled.reason == "target stale", stalled.reason


def test_control_loop_advances_the_clock_when_re_ticking_a_stale_frame():
    # End-to-end through the real control thread: one frame, then silence. The keepalive
    # must hand tick() a clock that keeps moving.
    seen = []
    fc = StubFC(switch_active=True, armed=True)

    class _OneFrameThenStall:
        def open(self): pass
        def close(self): pass
        def frames(self):
            yield _det_bundle(160.0)
            time.sleep(0.45)                       # camera has stalled

    pipe = Pipeline(_OneFrameThenStall(), IouAssociator(iou_threshold=0.2),
                    _servo(320, 240), _safety(), fc, display=lambda *a, **k: None)
    pipe._control_fallback_s = 0.05
    orig = pipe.tick
    pipe.tick = lambda b, now=None: (seen.append(now), orig(b, now))[1]
    pipe.run()

    advanced = [n for n in seen if n is not None]
    assert len(advanced) >= 3, f"expected keepalive re-ticks, got {seen}"
    assert advanced == sorted(advanced), "keepalive clock must be monotonic"
    assert advanced[-1] - advanced[0] > 0.15, f"clock barely moved: {advanced}"
