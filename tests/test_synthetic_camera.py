import numpy as np

from pi_fpv_companion.camera.synthetic import SyntheticCamera


def test_frame_has_correct_shape_and_dtype():
    cam = SyntheticCamera(width=720, height=576)
    b = cam.render_at(0.0)
    assert b.image.shape == (576, 720, 3)
    assert b.image.dtype == np.uint8


def test_one_detection_per_frame():
    cam = SyntheticCamera()
    b = cam.render_at(0.0)
    assert len(b.detections) == 1
    assert b.detections[0].class_name == "target"


def test_target_center_is_near_frame_center_at_t_zero():
    cam = SyntheticCamera(width=720, height=576)
    b = cam.render_at(0.0)
    d = b.detections[0]
    # sin(0)=0, so center == frame center at t=0
    assert d.x == 720 / 2
    assert d.y == 576 / 2


def test_target_moves_away_from_center_over_time():
    cam = SyntheticCamera(width=720, height=576)
    centers_x = [cam.render_at(t).detections[0].x for t in (0.0, 0.5, 1.0, 1.5)]
    # Should not all be at center
    assert max(abs(x - 360) for x in centers_x) > 10


def test_rendered_pixels_contain_red_at_target_location():
    cam = SyntheticCamera()
    b = cam.render_at(0.0)
    d = b.detections[0]
    cy, cx = int(d.y), int(d.x)
    # BGR red at the center pixel of the target
    assert tuple(b.image[cy, cx]) == (0, 0, 255)
