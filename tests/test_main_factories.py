"""Smoke tests for main.py's factory functions.

These tests verify the wiring from config -> Camera/Detector/Tracker/FC objects
without actually running the pipeline. They catch the kind of "config field is
silently ignored" bug the audit surfaced.
"""
from pathlib import Path

import pytest

from pi_fpv_companion.config import load
from pi_fpv_companion.main import (
    _build_camera,
    _build_detector,
    _build_fc,
    _build_tracker,
    _enforce_fc_params,
    _resolve_class_ids,
)


_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"


def test_resolve_class_ids_known_coco_names():
    ids = _resolve_class_ids(["person", "car", "boat"])
    assert ids == (0, 2, 8)


def test_resolve_class_ids_unknown_names_dropped():
    # Unknown names log a warning but don't crash; known ones come through.
    ids = _resolve_class_ids(["person", "definitely_not_a_class", "car"])
    assert 0 in ids
    assert 2 in ids
    assert len(ids) == 2


def test_resolve_class_ids_empty_input():
    assert _resolve_class_ids([]) == ()
    assert _resolve_class_ids(None) == ()


def test_mac_dev_config_builds_synthetic_camera():
    cfg = load(_CONFIG_ROOT / "mac-dev.yaml")
    cam = _build_camera(cfg)
    from pi_fpv_companion.camera.synthetic import SyntheticCamera
    assert isinstance(cam, SyntheticCamera)


def test_mac_dev_config_builds_iou_tracker():
    cfg = load(_CONFIG_ROOT / "mac-dev.yaml")
    tracker = _build_tracker(cfg)
    from pi_fpv_companion.track.iou_associator import IouAssociator
    assert isinstance(tracker, IouAssociator)


def test_mac_dev_config_builds_ardupilot_backend():
    cfg = load(_CONFIG_ROOT / "mac-dev.yaml")
    fc = _build_fc(cfg)
    from pi_fpv_companion.fc.ardupilot import ArduPilotBackend
    assert isinstance(fc, ArduPilotBackend)


def test_mac_dev_config_has_no_detector():
    cfg = load(_CONFIG_ROOT / "mac-dev.yaml")
    assert _build_detector(cfg) is None


def _tracker_cfg(tmp_path, tracker_line):
    p = tmp_path / "t.yaml"
    p.write_text(
        "camera: {type: synthetic}\ndetector: {type: none}\n"
        f"tracker: {{{tracker_line}}}\nfc: {{backend: ardupilot}}\n"
    )
    return load(p)


def test_classical_tracker_factory_builds_mosse(tmp_path):
    cfg = _tracker_cfg(tmp_path, "type: classical, cv2_backend: mosse")
    tracker = _build_tracker(cfg)
    from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker
    assert isinstance(tracker, ClassicalCv2Tracker)
    assert tracker.backend == "mosse"


def test_legacy_kcf_tracker_type_alias_still_works(tmp_path):
    """Old configs with `tracker.type: kcf` should still construct (classical + kcf backend)."""
    cfg = _tracker_cfg(tmp_path, "type: kcf")
    assert cfg.tracker.type == "classical"
    assert cfg.tracker.cv2_backend == "kcf"
    tracker = _build_tracker(cfg)
    from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker
    assert isinstance(tracker, ClassicalCv2Tracker)
    assert tracker.backend == "kcf"


def test_enforce_fc_params_builds_desired_set_from_config():
    # Startup validation must enforce ANGLE_MAX (= angle_max_deg×100) and the
    # companion's RC channels' *_OPTION=0, plus any explicit overrides.
    cfg = load(_CONFIG_ROOT / "imx500.yaml")        # switch ch7, select ch9, angle_max 45
    cfg.fc.enforce_params = {"SR2_EXTRA2": 5}

    class StubFC:
        def __init__(self): self.seen = None
        def ensure_params(self, desired): self.seen = desired; return {k: "ok" for k in desired}

    fc = StubFC()
    _enforce_fc_params(cfg, fc)
    assert fc.seen["ANGLE_MAX"] == 4500
    assert fc.seen["RC7_OPTION"] == 0               # companion mode switch
    assert fc.seen["RC9_OPTION"] == 0               # target-select channel
    assert fc.seen["SR2_EXTRA2"] == 5               # operator override


def test_enforce_fc_params_skipped_when_disabled():
    cfg = load(_CONFIG_ROOT / "imx500.yaml")
    cfg.fc.enforce_params_on_start = False

    class StubFC:
        def __init__(self): self.called = False
        def ensure_params(self, desired): self.called = True; return {}

    fc = StubFC()
    _enforce_fc_params(cfg, fc)
    assert fc.called is False


def test_imx500_detector_is_none_sensor_does_inference(tmp_path):
    # On the IMX500 path the detector is None — the sensor emits detections inline.
    cfg = load(_CONFIG_ROOT / "imx500.yaml")
    assert _build_detector(cfg) is None
