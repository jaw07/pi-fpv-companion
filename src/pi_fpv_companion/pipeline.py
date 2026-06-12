"""Main loop: camera -> [detector] -> tracker -> visual servo -> safety -> FC backend.

The detector is OPTIONAL because the flight camera (IMX500) and the dev
SyntheticCamera emit detections inline in the FrameBundle. When the camera does
that, Pipeline leaves them alone. When it doesn't (File, Webcam) and a detector
is configured, Pipeline runs it inline on the configured cadence.

Generic over camera/detector/tracker/FC implementations. Same Pipeline runs in
dev (SyntheticCamera + FakeArduCopter) and production (IMX500 + ArduPilotBackend
over UART).
"""
from __future__ import annotations
import time
from dataclasses import replace
from typing import Callable, Optional

from pi_fpv_companion.camera.base import Camera, FrameBundle
from pi_fpv_companion.detect.base import Detector
from pi_fpv_companion.guidance.safety import GateResult, SafetyConfig, gate
from pi_fpv_companion.guidance.visual_servo import ClosureState, DiveState, ServoConfig, compute_intent
from pi_fpv_companion.guidance.rate_control import RateConfig, RateState, compute_rate_intent
from pi_fpv_companion.track.base import Tracker
from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState, Target, ZERO_INTENT
from typing import List

StatusCallback = Callable[
    [Optional[Target], GuidanceIntent, GateResult, SwitchState, bool, FrameBundle,
     Optional[List[Target]]], None
]

# STANDBY-in-GUIDED_NOGPS levelling bridge: on the engaged->STANDBY edge the FC may be
# coasting on a committed dive attitude with the pilot's sticks dead (guided ignores
# them), so the companion levels it for this long — then goes COMPLETELY SILENT.
# After our last setpoint, ArduCopter's own guided timeout (GUID_TIMEOUT, default 3 s,
# enforced from imx500.yaml) zeroes the lean angles and holds zero climb rate natively.
# Net: ~2 s of our levelling + the FC's own hold, and steady-state STANDBY injects
# nothing — even with the FC left in GUIDED_NOGPS.
_STANDBY_GUIDED_BRIDGE_S = 2.0


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
        on_status: Optional[StatusCallback] = None,
        force_mode: Optional[GuidanceMode] = None,
        camera_watchdog_s: float = 0.0,
        first_frame_grace_s: float = 15.0,
        rate_cfg: Optional[RateConfig] = None,
    ) -> None:
        self._force_mode = force_mode
        # Camera stall watchdog (0 = off; production sets ~2s). The IMX500/
        # libcamera frontend can hang with capture_request() blocking the main
        # loop forever — a separate thread is the only way to recover.
        self._camera_watchdog_s = camera_watchdog_s
        # How long to wait for the FIRST frame after (re)start before forcing a
        # restart. Must exceed a clean cold open (~5-6s: rpk upload to the sensor)
        # but short enough that a hung reopen recovers fast. 15s = ~2.5x margin.
        self._first_frame_grace_s = first_frame_grace_s
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
        # Operator target selection (multi-target tracker only): a rising edge on
        # the FC's select channel cycles the lock among current detections.
        self._last_select_pwm = 0
        # ch7 auto-engage: track the STANDBY<->engaged edge so we command the FC's
        # flight mode (fc.set_engaged) only on the transition, never every frame.
        self._last_engaged = False
        # STANDBY-in-guided levelling bridge deadline (see _STANDBY_GUIDED_BRIDGE_S).
        self._standby_bridge_until = 0.0
        self._tracks: Optional[list] = None
        self._dive_entered_t: Optional[float] = None   # for the DIVE lean soft-start
        self._closure = ClosureState()                  # TRACK PI closure integrator
        self._dive = DiveState()                        # DIVE lean low-pass (anti-nod)
        # guided_nogps body-RATE path (control_mode: guided_nogps). None -> the STABILIZE/RC
        # path below is used unchanged. When set, step() dispatches to _step_rate instead.
        self._rate_cfg = rate_cfg
        self._rate_state = RateState()
        self._last_rate_mode: Optional[GuidanceMode] = None

        # Alpha-beta filter + wrong-target gating sits between the raw tracker
        # and the servo/safety. Everything downstream consumes FilteredTarget.
        from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
        self._target_filter = AlphaBetaTargetFilter()

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
        grace = max(self._first_frame_grace_s, self._camera_watchdog_s)

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

        # Use the camera's intrinsic detections if it produced any (IMX500 emits them
        # inline); otherwise run the configured detector inline on the scheduled cadence.
        detections = list(bundle.detections)
        if not detections and self._detector is not None:
            if self._frame_idx % self._detect_period == 0:
                detections = self._detector.detect(bundle.image)

        switch = self._fc.read_switch()
        if self._force_mode is not None:
            switch = replace(switch, mode=self._force_mode,
                             active=self._force_mode is not GuidanceMode.STANDBY)
        armed = self._fc.is_armed()

        # ch7 auto-engage: on the STANDBY<->engaged edge, tell the FC to switch into
        # (or restore from) the control_mode's flight mode. No-op unless the backend
        # has auto_guided enabled; the control_ready interlock below still gates any
        # actual command-sending until the FC confirms it is in that mode.
        engaged = switch.mode is not GuidanceMode.STANDBY
        if engaged is not self._last_engaged:
            set_engaged = getattr(self._fc, "set_engaged", None)
            if callable(set_engaged):
                set_engaged(engaged)
            self._last_engaged = engaged

        # Operator target selection (multi-target tracker): a rising edge on the FC
        # select channel (ch8) cycles the locked target among the current
        # detections. Allowed ONLY in STANDBY — you choose your target before
        # committing; once engaged (TRACK/DIVE) the lock is FROZEN so a stray
        # ch8 bump can't swap targets mid-engagement. The lock persists across the
        # mode switch, so what you pick in STANDBY stays locked through TRACK/DIVE.
        cycle_fn = getattr(self._tracker, "cycle", None)
        sel_fn = getattr(self._fc, "select_pwm", None)
        if callable(cycle_fn) and callable(sel_fn):
            pwm = sel_fn()
            if (switch.mode is GuidanceMode.STANDBY
                    and pwm >= 1700 and self._last_select_pwm < 1700):
                cycle_fn()
            self._last_select_pwm = pwm
            # Acquire/re-acquire only in STANDBY; once committed, a dropped target
            # holds (no silent swap to a different target — see MultiObjectTracker).
            if hasattr(self._tracker, "auto_acquire"):
                self._tracker.auto_acquire = switch.mode is GuidanceMode.STANDBY

        raw_target = self._tracker.consume(bundle.image, detections, now)
        self._tracks = getattr(self._tracker, "tracks", None)   # all tracks for the HUD

        # Filter + quality-assess. Everything downstream uses the FilteredTarget,
        # never the raw tracker output (audit §4/§5).
        target = self._target_filter.update(
            raw_target, bundle.width, bundle.height, now
        )
        # guided_nogps body-RATE path: dispatch to the rate controller and return early; the
        # STABILIZE/RC-override path below is left completely unchanged for other control_modes.
        if self._rate_cfg is not None:
            return self._tick_rate(switch, target, armed, now, bundle)
        if target is not None:
            # Preview the intent even in STANDBY (using TRACK behaviour) so the
            # HUD shows what guidance would do; the gate decides what's actually
            # sent. When engaged, the switch mode (TRACK/DIVE) drives closure.
            preview_mode = (
                switch.mode if switch.mode is not GuidanceMode.STANDBY else GuidanceMode.TRACK
            )
            # Time since DIVE was engaged, for the lean soft-start (reset on exit).
            # The DiveState (lean low-pass) is active only while actually in DIVE.
            if switch.mode is GuidanceMode.DIVE:
                if self._dive_entered_t is None:
                    self._dive_entered_t = now
                dive_elapsed_s = now - self._dive_entered_t
                dive = self._dive
            else:
                self._dive_entered_t = None
                dive_elapsed_s = 1e9
                self._dive.reset()
                dive = None
            # PI closure integrator: active only when actually engaged in TRACK.
            # STANDBY (preview-as-TRACK) and DIVE must not wind it, so reset + skip.
            if switch.mode is GuidanceMode.TRACK:
                closure = self._closure
            else:
                self._closure.reset()
                closure = None
            # Measured airframe pitch for the DIVE pitch-fold (agnostic dive). Backends
            # that don't report attitude (or before the first ATTITUDE msg) -> 0, which
            # makes the fold inert (frame-only homing).
            pitch_fn = getattr(self._fc, "pitch_deg", None)
            pitch_meas = pitch_fn() if pitch_fn is not None else 0.0
            roll_fn = getattr(self._fc, "roll_deg", None)
            roll_meas = roll_fn() if roll_fn is not None else 0.0
            intent = compute_intent(target, self._servo_cfg, preview_mode,
                                    dive_elapsed_s, closure, dive,
                                    pitch_deg_measured=pitch_meas,
                                    roll_deg_measured=roll_meas)
        else:
            intent = ZERO_INTENT
        gated = gate(intent, target, switch, armed, now, self._safety_cfg)
        # STANDBY -> release to the pilot's radio. Engaged -> only override if the
        # FC is in the flight mode our control_mode expects (control_ready
        # interlock); otherwise release, so we never push sticks into the wrong
        # mode. (When the gate mutes while engaged, gated.intent is neutral -> hold.)
        # ALSO release while disarmed: ZERO_INTENT's thrust is HOVER (0.5), which in
        # stabilize maps to the hover throttle PWM — a standing throttle-at-hover
        # override on a disarmed FC self-launches the craft at arm (same flight-2
        # failure class as the rate path's hover-hold; gate() mutes the intent but
        # muting yields ZERO_INTENT, not silence, so the disarm case must release).
        can_command = armed or not self._safety_cfg.require_armed
        ready = getattr(self._fc, "control_ready", None)
        if (switch.mode is GuidanceMode.STANDBY or not can_command
                or (ready is not None and not ready())):
            self._fc.release()
        else:
            self._fc.send_intent(gated.intent)

        if self._on_status is not None:
            self._on_status(target, intent, gated, switch, armed, bundle, self._tracks)

        return gated

    def _tick_rate(self, switch, target, armed, now, bundle) -> GateResult:
        """guided_nogps body-RATE dispatch (mode-aware: TRACK follows + holds range/altitude,
        DIVE commits). STANDBY / not-control_ready -> release. Reuses the safety gate() for
        staleness/quality/armed; the impact STOP is sent regardless (it IS the safe action at
        ground contact). Resets the rate state on any mode change."""
        import math as _math
        fc = self._fc
        if switch.mode is not self._last_rate_mode:
            self._rate_state.reset()
            self._last_rate_mode = switch.mode
            if switch.mode is GuidanceMode.STANDBY:
                # Entering STANDBY: arm the levelling bridge (see _STANDBY_GUIDED_BRIDGE_S).
                self._standby_bridge_until = now + _STANDBY_GUIDED_BRIDGE_S
        ready = getattr(fc, "control_ready", None)
        fc_in_guided = ready is None or ready()   # FC actually in GUIDED_NOGPS (accepts our rates)
        # NEVER command body rates while disarmed (unless the config waives the armed
        # gate for bench work). A standing SET_ATTITUDE_TARGET with hover thrust while
        # disarmed on the ground means the craft launches itself the instant the pilot
        # arms — flight-2 finding. The gate() below also enforces this when engaged,
        # but the STANDBY hover-hold branch bypasses gate(), so check it explicitly.
        can_command = armed or not self._safety_cfg.require_armed
        if switch.mode is GuidanceMode.STANDBY or not fc_in_guided:
            fc.release()                            # clear any stray RC override (no-op in guided)
            if (fc_in_guided and can_command and hasattr(fc, "send_body_rates")
                    and now < self._standby_bridge_until):
                # Engaged->STANDBY with the FC STILL in GUIDED_NOGPS (sticks dead there by
                # ArduPilot design): the FC may be coasting on a committed dive attitude, so
                # LEVEL it for the short bridge window — then go silent and let the FC's own
                # GUID_TIMEOUT hold (level + zero climb) take over. STANDBY must not inject
                # commands at steady state, even in the flight mode. Manual recovery is
                # unchanged: the pilot flips the FC mode OUT of GUIDED_NOGPS at any time.
                p = _math.radians(fc.pitch_deg()) if hasattr(fc, "pitch_deg") else 0.0
                r = _math.radians(fc.roll_deg()) if hasattr(fc, "roll_deg") else 0.0
                if hasattr(fc, "climb_mps"):
                    self._rate_state.hover = max(0.05, min(0.6, self._rate_state.hover - 0.01 * fc.climb_mps()))
                fc.send_body_rates(max(-0.6, min(0.6, 2.0 * (0.0 - r))),
                                   max(-0.6, min(0.6, 2.0 * (0.0 - p))),
                                   0.0, self._rate_state.hover)
                reason = "standby-bridge (levelling, then silent)"
            elif fc_in_guided and can_command:
                reason = "standby (silent; FC guided-timeout holds)"
            elif fc_in_guided:
                reason = "standby (disarmed; no commands)"
            else:
                reason = "manual (FC not in GUIDED_NOGPS)"
            gated = GateResult(ZERO_INTENT, True, reason)
            if self._on_status is not None:
                self._on_status(target, ZERO_INTENT, gated, switch, armed, bundle, self._tracks)
            return gated
        if not can_command:
            # Engaged but disarmed: send NOTHING, and keep the controller state
            # pristine — running the rate controller here would wind its integrals /
            # dive timers against a craft that can't move, releasing them as a step
            # input on the first armed tick (the launch-at-arm class again, via state).
            self._rate_state.reset()
            gated = GateResult(ZERO_INTENT, True, "fc not armed")
            if self._on_status is not None:
                self._on_status(target, ZERO_INTENT, gated, switch, armed, bundle, self._tracks)
            return gated
        pitch = _math.radians(fc.pitch_deg()) if hasattr(fc, "pitch_deg") else 0.0
        roll = _math.radians(fc.roll_deg()) if hasattr(fc, "roll_deg") else 0.0
        gamma = fc.flight_path_angle_rad() if hasattr(fc, "flight_path_angle_rad") else 0.0
        agl = fc.agl_m() if hasattr(fc, "agl_m") else 1e9
        # Online hover trim: TRACK holds altitude, so nudge the hover thrust toward null climb
        # (TWR-independent; the high-TWR airframe hovers well below 0.5).
        # Only learn hover while roughly LEVEL: climb then reflects hover error, not a commanded
        # lean/descent. Trimming during a pitched-down chase would crank hover up (the craft is
        # sinking by intent), then a subsequent SEARCH/hold would balloon up on that bad hover.
        if (switch.mode is GuidanceMode.TRACK and target is not None
                and hasattr(fc, "climb_mps") and abs(pitch) < 0.26):
            self._rate_state.hover = max(0.05, min(0.6, self._rate_state.hover - 0.01 * fc.climb_mps()))
        ri = compute_rate_intent(target, self._rate_cfg, self._rate_state, now, mode=switch.mode,
                                 pitch_rad=pitch, roll_rad=roll, gamma_rad=gamma, agl_m=agl)
        # Safety gate (armed / staleness / quality). Probe carries the commanded thrust+yaw.
        probe = GuidanceIntent(0.0, 0.0, _math.degrees(ri.yaw_rate), ri.thrust,
                               target.timestamp if target is not None else now)
        gated = gate(probe, target, switch, armed, now, self._safety_cfg)
        if ri.phase == "STOP" or not gated.muted:
            fc.send_body_rates(ri.roll_rate, ri.pitch_rate, ri.yaw_rate, ri.thrust)
        else:
            # Muted (no/stale/low-quality target) -> SAFE HOLD: level + hover, never search-dive.
            fc.send_body_rates(0.0, max(-0.6, min(0.6, 2.0 * (0.0 - pitch))), 0.0, self._rate_state.hover)
        if self._on_status is not None:
            hud = GuidanceIntent(0.0, 0.0, _math.degrees(ri.yaw_rate), ri.thrust, now)
            self._on_status(target, hud, gated, switch, armed, bundle, self._tracks)
        return gated
