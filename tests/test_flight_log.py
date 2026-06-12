"""FlightRecorder (companion blackbox) unit tests: record content, rate limiting,
size rotation + pruning, and fail-open behavior (recording must never block flight)."""
from __future__ import annotations
import json

from pi_fpv_companion.flight_log import FlightRecorder
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import (
    Detection, FilteredTarget, GuidanceMode, SwitchState, ZERO_INTENT)


def _switch(mode=GuidanceMode.TRACK, pwm=1500):
    return SwitchState(active=mode is not GuidanceMode.STANDBY, pwm_us=pwm,
                       timestamp=0.0, mode=mode)


def _target():
    det = Detection(x=100, y=120, w=40, h=60, confidence=0.9, class_id=0, class_name="person")
    return FilteredTarget(detection=det, track_id=3, vx_px_s=0.0, vy_px_s=0.0,
                          quality=0.8, timestamp=1.0, measurement_timestamp=1.0)


def _gated(muted=False, reason=""):
    return GateResult(ZERO_INTENT, muted, reason)


def test_records_decision_trail_as_jsonl(tmp_path):
    rec = FlightRecorder(tmp_path, rate_hz=1000.0)
    rec.record(_target(), ZERO_INTENT, _gated(muted=True, reason="fc not armed"),
               _switch(GuidanceMode.DIVE, 1800), armed=False)
    rec.close()
    lines = [json.loads(ln) for f in tmp_path.glob("flight-*.jsonl")
             for ln in f.read_text().splitlines()]
    assert len(lines) == 1
    r = lines[0]
    assert r["mode"] == "DIVE" and r["pwm"] == 1800 and r["armed"] is False
    assert r["muted"] is True and r["reason"] == "fc not armed"
    assert r["tgt"] == {"x": 100, "y": 120, "w": 40, "h": 60, "q": 0.8, "id": 3}


def test_rate_limited_and_no_target_is_omitted(tmp_path):
    rec = FlightRecorder(tmp_path, rate_hz=1.0)     # 1 Hz: 2nd immediate call is dropped
    rec.record(None, ZERO_INTENT, _gated(), _switch(), armed=True)
    rec.record(None, ZERO_INTENT, _gated(), _switch(), armed=True)
    rec.close()
    lines = [json.loads(ln) for f in tmp_path.glob("flight-*.jsonl")
             for ln in f.read_text().splitlines()]
    assert len(lines) == 1
    assert "tgt" not in lines[0]


def test_rotates_by_size_and_prunes_oldest(tmp_path):
    rec = FlightRecorder(tmp_path, rate_hz=0.0, max_bytes=120, keep_files=2)
    for _ in range(10):                              # each line > 60 B -> several rotations
        rec.record(_target(), ZERO_INTENT, _gated(), _switch(), armed=True)
    rec.close()
    files = sorted(tmp_path.glob("flight-*.jsonl"))
    assert len(files) <= 3                           # keep_files + the active file


def test_unwritable_directory_disables_quietly(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the directory should be")
    rec = FlightRecorder(blocker / "flight", rate_hz=1000.0)   # mkdir fails -> disabled
    rec.record(_target(), ZERO_INTENT, _gated(), _switch(), armed=True)   # must not raise
    rec.close()
