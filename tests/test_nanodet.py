"""Tests for parts of NanoDetDetector that don't need a model file —
anchor generation and the letterbox preprocess. The full forward pass + decode
is exercised by `scripts/profile_nanodet.py` against a real model."""
import math
from pathlib import Path

import numpy as np

from pi_fpv_companion.detect.nanodet import NanoDetConfig, NanoDetDetector


def _cfg(input_size=320):
    return NanoDetConfig(
        model_dir=Path("/nonexistent"),       # not loaded — open() not called
        input_size=input_size,
    )


def test_anchor_count_matches_expected_total():
    cfg = _cfg(input_size=320)
    det = NanoDetDetector(cfg)
    # Anchor count = sum over strides of (ceil(input/stride))^2
    expected = sum(math.ceil(320 / s) ** 2 for s in cfg.strides)
    assert det._anchors.shape == (expected, 2)


def test_anchor_centers_align_to_grid_for_smallest_stride():
    cfg = _cfg(input_size=320)
    det = NanoDetDetector(cfg)
    # First N anchors belong to stride 8; first row should be at y = 8/2 = 4
    stride = 8
    fw = math.ceil(320 / stride)
    first_row = det._anchors[:fw]
    assert (first_row[:, 1] == stride / 2).all()
    # x positions should be stride/2, stride/2 + stride, ...
    expected_x = np.arange(fw) * stride + stride / 2
    assert np.allclose(first_row[:, 0], expected_x)


def test_letterbox_preserves_aspect_ratio():
    cfg = _cfg(input_size=320)
    det = NanoDetDetector(cfg)
    img = np.zeros((576, 720, 3), dtype=np.uint8)
    letterboxed, scale, pad_w, pad_h = det._letterbox_preprocess(img)
    assert letterboxed.shape == (320, 320, 3)
    # Wider-than-tall input: should be scaled to fit width
    assert scale == 320 / 720
    # Pads top + bottom
    new_h = int(576 * scale)
    expected_pad_h = (320 - new_h) // 2
    assert pad_h == expected_pad_h
    assert pad_w == 0


def test_stride_per_anchor_partitions_into_correct_buckets():
    cfg = _cfg(input_size=320)
    det = NanoDetDetector(cfg)
    strides_arr = det._stride_per_anchor()
    # Total length matches anchors
    assert strides_arr.shape[0] == det._anchors.shape[0]
    # First bucket is stride 8
    fw = math.ceil(320 / 8)
    first_count = fw * fw
    assert (strides_arr[:first_count] == 8).all()
    assert strides_arr[first_count] != 8     # next bucket is a different stride
