import numpy as np

from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.detect.color import ColorBlobDetector


def test_finds_red_target_in_synthetic_frame():
    cam = SyntheticCamera(width=320, height=240, target_size_px=40)
    bundle = cam.render_at(0.0)
    det = ColorBlobDetector(min_area_px=100)
    found = det.detect(bundle.image)
    assert len(found) == 1
    # Center of the red square is the frame center at t=0
    assert abs(found[0].x - 160) < 5
    assert abs(found[0].y - 120) < 5


def test_returns_no_detections_for_uniform_image():
    img = np.full((240, 320, 3), 64, dtype=np.uint8)  # dark gray everywhere
    det = ColorBlobDetector(min_area_px=100)
    assert det.detect(img) == []


def test_min_area_filters_out_speckle():
    img = np.full((240, 320, 3), 64, dtype=np.uint8)
    img[100:102, 100:102] = (0, 0, 255)  # 2x2 red speck
    det = ColorBlobDetector(min_area_px=100)
    assert det.detect(img) == []
