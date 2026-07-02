"""STANDBY safety-contract verifier (pure, testable).

The flight-2 fixes (PR #36) gave the companion a hard command contract:

  - STANDBY (engage switch below track_threshold): the companion injects NOTHING
    that touches flight control — no non-zero RC_CHANNELS_OVERRIDE, no
    SET_ATTITUDE_TARGET, no DO_SET_MODE. (An all-zero override = "hand back to the
    pilot" is allowed, and only as a short burst.)
  - DISARMED: no SET_ATTITUDE_TARGET and no non-zero override, in ANY switch state.
  - DO_SET_MODE: only on a STANDBY->engaged (or engaged->STANDBY restore) edge,
    never in steady-state STANDBY.

That contract is verified in unit tests and SITL, but the highest-confidence
proof is to watch the ACTUAL MAVLink wire on the real FC. This module is the pure
verifier: feed it the time-ordered stream of observed MAVLink events (switch PWM,
armed state, and each companion->FC command) and it reports every contract
violation. `scripts/check_wire_contract.py` is the thin MAVLink/tlog reader that
drives it; the logic here is hardware-free and unit-tested.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ContractConfig:
    switch_channel: int = 7
    track_threshold_us: int = 1300   # pwm >= this -> engaged (TRACK/DIVE), else STANDBY
    # An all-zero override burst right after disengage is allowed (the "hand back"
    # instruction); more than this many consecutive override frames in STANDBY is a
    # steady-state leak (the contract says STANDBY goes radio-silent after the burst).
    release_burst_max: int = 12
    # A DO_SET_MODE within this many STANDBY frames of leaving engaged is the legit
    # restore-on-disengage edge; beyond it, a mode command is a settled-STANDBY breach.
    standby_edge_frames: int = 3


@dataclass(frozen=True)
class Violation:
    t: float
    kind: str        # human-readable contract that was broken
    detail: str


@dataclass
class ContractChecker:
    """Feed events in time order; collects violations. State machine mirrors the
    pipeline's command contract so a violation here = a real wire-level breach."""
    cfg: ContractConfig = field(default_factory=ContractConfig)
    violations: List[Violation] = field(default_factory=list)
    _armed: bool = False
    _armed_known: bool = False
    _pwm: int = 0
    _pwm_known: bool = False
    _override_run: int = 0          # consecutive override frames in the current STANDBY stretch
    _standby_run: int = 0           # consecutive STANDBY rc_channels frames (0 = just engaged/edge)
    counts: dict = field(default_factory=lambda: {
        "heartbeat": 0, "rc_channels": 0, "override_zero": 0, "override_nonzero": 0,
        "attitude_target": 0, "set_mode": 0})

    # ---- state inputs ----
    def on_heartbeat(self, t: float, armed: bool) -> None:
        self._armed, self._armed_known = armed, True
        self.counts["heartbeat"] += 1

    def on_rc_channels(self, t: float, switch_pwm: int) -> None:
        self._pwm, self._pwm_known = switch_pwm, True
        self.counts["rc_channels"] += 1
        if self._standby():
            self._standby_run += 1    # count how long we've been settled in STANDBY
        else:
            self._override_run = 0    # engaged: reset the STANDBY burst counter
            self._standby_run = 0     # ...and the STANDBY-settled counter (this is an edge)

    # ---- companion -> FC commands ----
    def on_rc_override(self, t: float, channels: List[int]) -> None:
        nonzero = any(c for c in channels)
        if nonzero:
            self.counts["override_nonzero"] += 1
            if self._standby():
                self._add(t, "STANDBY-no-override",
                          f"non-zero RC_CHANNELS_OVERRIDE in STANDBY: {channels}")
            if self._armed_known and not self._armed:
                self._add(t, "DISARMED-no-override",
                          f"non-zero RC_CHANNELS_OVERRIDE while DISARMED: {channels}")
        else:
            self.counts["override_zero"] += 1
            if self._standby():
                self._override_run += 1
                if self._override_run > self.cfg.release_burst_max:
                    self._add(t, "STANDBY-radio-silence",
                              f"{self._override_run} consecutive override frames in "
                              "STANDBY — should burst then go silent")

    def on_attitude_target(self, t: float) -> None:
        self.counts["attitude_target"] += 1
        if self._standby():
            self._add(t, "STANDBY-no-attitude-target",
                      "SET_ATTITUDE_TARGET in STANDBY")
        if self._armed_known and not self._armed:
            self._add(t, "DISARMED-no-attitude-target",
                      "SET_ATTITUDE_TARGET while DISARMED")

    def on_set_mode(self, t: float, mode: int) -> None:
        self.counts["set_mode"] += 1
        # DO_SET_MODE is legitimate only on an engage/disengage EDGE. We approximate the
        # edge with _standby_run: a mode command at the restore edge lands within a frame
        # or two of leaving engaged (_standby_run small). A mode command once the switch
        # has SETTLED in STANDBY (armed OR disarmed) is the flight-3 hijack signature —
        # the FC being commanded to GUIDED_NOGPS while the pilot believes he's in STANDBY.
        # The previous check only caught the disarmed case, so an ARMED in-flight hijack
        # passed. Now both fail. _STANDBY_EDGE_FRAMES tolerates the genuine restore edge.
        if self._standby() and self._standby_run > self.cfg.standby_edge_frames:
            armed_s = "armed" if self._armed else ("disarmed" if self._armed_known else "unknown-arm")
            self._add(t, "STANDBY-no-mode-cmd",
                      f"DO_SET_MODE({mode}) in settled STANDBY ({armed_s}, "
                      f"{self._standby_run} frames since engaged)")

    # ---- helpers ----
    def _standby(self) -> bool:
        return self._pwm_known and self._pwm < self.cfg.track_threshold_us

    def _add(self, t: float, kind: str, detail: str) -> None:
        self.violations.append(Violation(t, kind, detail))

    def report(self) -> str:
        lines = ["=== STANDBY safety-contract report ===",
                 f"observed: {self.counts}"]
        if not self.violations:
            lines.append("RESULT: PASS — no contract violations observed.")
        else:
            lines.append(f"RESULT: FAIL — {len(self.violations)} violation(s):")
            for v in self.violations[:50]:
                lines.append(f"  [{v.t:.3f}] {v.kind}: {v.detail}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        return not self.violations
