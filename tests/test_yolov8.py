"""Tests for the YOLOv8 detector's decoder. The forward pass requires a real
NCNN model; profile that via `scripts/profile_nanodet.py` after wiring a
similar profile script for YOLOv8."""
from pathlib import Path

import numpy as np

from pi_fpv_companion.detect.yolov8 import Yolov8Config, Yolov8Detector


def _cfg(input_size=256):
    return Yolov8Config(model_dir=Path("/nonexistent"), input_size=input_size)


def test_letterbox_preserves_aspect_ratio():
    det = Yolov8Detector(_cfg(input_size=256))
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    letterboxed, scale, pad_w, pad_h = det._letterbox_preprocess(img)
    assert letterboxed.shape == (256, 256, 3)
    assert scale == 256 / 640
    new_h = int(480 * scale)
    assert pad_h == (256 - new_h) // 2
    assert pad_w == 0


def test_decode_empty_output_returns_no_detections():
    det = Yolov8Detector(_cfg())
    # Below threshold everywhere
    n_anchors = 1344                              # arbitrary
    output = np.zeros((84, n_anchors), dtype=np.float32)
    out = det._decode(output, scale=1.0, pad_w=0, pad_h=0, orig_w=256, orig_h=256)
    assert out == []


def test_decode_finds_single_high_confidence_detection():
    cfg = _cfg(input_size=256)
    det = Yolov8Detector(cfg)
    n_anchors = 1344
    output = np.zeros((84, n_anchors), dtype=np.float32)
    # Plant one detection at anchor 0: cx=128, cy=64, w=40, h=40, class 0 confidence 0.9
    output[0, 0] = 128.0       # cx
    output[1, 0] = 64.0        # cy
    output[2, 0] = 40.0        # w
    output[3, 0] = 40.0        # h
    output[4, 0] = 0.9         # class 0 score

    dets = det._decode(output, scale=1.0, pad_w=0, pad_h=0, orig_w=256, orig_h=256)
    assert len(dets) == 1
    assert abs(dets[0].x - 128) < 1
    assert abs(dets[0].y - 64) < 1
    assert abs(dets[0].w - 40) < 1
    assert dets[0].confidence > 0.85
    assert dets[0].class_id == 0


def test_decode_unprojects_letterbox_pad_and_scale():
    cfg = _cfg(input_size=256)
    det = Yolov8Detector(cfg)
    n_anchors = 1344
    output = np.zeros((84, n_anchors), dtype=np.float32)
    # Detection at center of letterboxed input frame
    output[:4, 0] = [128.0, 128.0, 40.0, 40.0]
    output[4, 0] = 0.9
    # Letterbox parameters: original frame was 640x320 -> scale 0.4, padded to 256x256
    # Actually for 640x320 -> 256x128 box centered, pad_h=64, pad_w=0
    scale = 256 / 640
    pad_w = 0
    pad_h = (256 - int(320 * scale)) // 2

    dets = det._decode(output, scale=scale, pad_w=pad_w, pad_h=pad_h, orig_w=640, orig_h=320)
    assert len(dets) == 1
    # Center of input 256x256 maps back near center of 640x320 original
    assert abs(dets[0].x - 320) < 5
    assert abs(dets[0].y - 160) < 5


def test_decode_filters_below_conf_threshold():
    cfg = _cfg()
    cfg = Yolov8Config(**{**cfg.__dict__, "conf_threshold": 0.5})
    det = Yolov8Detector(cfg)
    n_anchors = 1344
    output = np.zeros((84, n_anchors), dtype=np.float32)
    output[:4, 0] = [128.0, 128.0, 40.0, 40.0]
    output[4, 0] = 0.3            # below 0.5

    dets = det._decode(output, scale=1.0, pad_w=0, pad_h=0, orig_w=256, orig_h=256)
    assert dets == []


def test_decode_rejects_target_class_mismatch():
    cfg = Yolov8Config(
        model_dir=Path("/nonexistent"),
        input_size=256,
        target_class_ids=(5,),            # only class 5
    )
    det = Yolov8Detector(cfg)
    n_anchors = 1344
    output = np.zeros((84, n_anchors), dtype=np.float32)
    # High-conf detection on class 0, which isn't in the target set
    output[:4, 0] = [128.0, 128.0, 40.0, 40.0]
    output[4, 0] = 0.9

    dets = det._decode(output, scale=1.0, pad_w=0, pad_h=0, orig_w=256, orig_h=256)
    assert dets == []
