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


def test_default_config_builds_mosse_classical_tracker():
    cfg = load(_CONFIG_ROOT / "default.yaml")
    tracker = _build_tracker(cfg)
    from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker
    assert isinstance(tracker, ClassicalCv2Tracker)
    assert tracker.backend == "mosse"


def test_legacy_kcf_tracker_type_alias_still_works():
    """Old configs with `tracker.type: kcf` should still construct."""
    import yaml
    from pi_fpv_companion.config import load
    cfg_path = _CONFIG_ROOT / "default.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    raw["tracker"]["type"] = "kcf"
    raw["tracker"].pop("cv2_backend", None)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        f.flush()
        cfg2 = load(f.name)
    assert cfg2.tracker.type == "classical"
    assert cfg2.tracker.cv2_backend == "kcf"
    tracker = _build_tracker(cfg2)
    from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker
    assert isinstance(tracker, ClassicalCv2Tracker)
    assert tracker.backend == "kcf"


def test_classes_of_interest_propagates_to_nanodet_config():
    """The bug the audit caught: classes_of_interest from YAML should flow into
    the detector's target_class_ids tuple. Construct NanoDetDetector and verify."""
    cfg = load(_CONFIG_ROOT / "default.yaml")
    # Don't actually call .open() (no model file present), but construction is enough
    # to verify the config got threaded through.
    det = _build_detector(cfg)
    from pi_fpv_companion.detect.nanodet import NanoDetDetector
    assert isinstance(det, NanoDetDetector)
    expected_ids = _resolve_class_ids(cfg.detector.classes_of_interest)
    assert det._cfg.target_class_ids == expected_ids
    assert 0 in det._cfg.target_class_ids       # person from YAML


def test_default_config_class_ids_resolve_to_known_coco_ids():
    """Verify the actual list of class ids is what we'd expect for the YAML names."""
    cfg = load(_CONFIG_ROOT / "default.yaml")
    ids = _resolve_class_ids(cfg.detector.classes_of_interest)
    assert 0 in ids   # person
    assert 2 in ids   # car
    assert 8 in ids   # boat
    # Should NOT contain ids for classes we didn't ask for
    assert 14 not in ids   # bird
