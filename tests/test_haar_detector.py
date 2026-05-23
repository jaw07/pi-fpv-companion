"""Smoke tests for HaarFaceDetector. We don't bundle a face image as a test asset,
so we can't validate detection accuracy here — that's manual via `demo_webcam.py`.
What we verify: the detector loads OpenCV's bundled cascade, accepts BGR images,
and returns a proper list of Detection objects (empty when no face is present).
"""
import numpy as np
import pytest

from pi_fpv_companion.detect.haar import HaarFaceDetector


def test_loads_bundled_cascade():
    det = HaarFaceDetector()
    # If construction succeeds the cascade loaded — the constructor raises otherwise
    assert det is not None


def test_returns_empty_list_on_uniform_image():
    det = HaarFaceDetector()
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    out = det.detect(img)
    assert isinstance(out, list)
    assert out == []


def test_returns_empty_list_on_random_noise():
    det = HaarFaceDetector(min_size_px=60)
    rng = np.random.default_rng(seed=42)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    out = det.detect(img)
    # Random noise virtually never trips Haar at min_size=60. Allow up to 1 false positive.
    assert len(out) <= 1


def test_downscale_returns_coordinates_in_original_frame():
    """When downscale=0.5 we detect on the half-res image but the returned bbox
    coordinates must be in the full-res frame's pixel space. Test by feeding an
    image that won't trigger detection and verifying the path doesn't crash."""
    det = HaarFaceDetector(downscale=0.5)
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert det.detect(img) == []
