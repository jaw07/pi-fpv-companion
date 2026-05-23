from unittest import mock

from pi_fpv_companion.cpu_affinity import compute_split, pin_current_thread


def test_disabled_returns_none():
    assert compute_split(False) == (None, None)


def test_four_cores_split_lower_upper():
    with mock.patch("os.cpu_count", return_value=4), \
         mock.patch("os.sched_setaffinity", create=True):
        pipe, det = compute_split(True)
    assert pipe == {0, 1}
    assert det == {2, 3}
    assert pipe.isdisjoint(det)


def test_eight_cores_split():
    with mock.patch("os.cpu_count", return_value=8), \
         mock.patch("os.sched_setaffinity", create=True):
        pipe, det = compute_split(True)
    assert pipe == {0, 1, 2, 3}
    assert det == {4, 5, 6, 7}


def test_fewer_than_four_cores_skips():
    with mock.patch("os.cpu_count", return_value=2), \
         mock.patch("os.sched_setaffinity", create=True):
        assert compute_split(True) == (None, None)


def test_no_syscall_platform_skips():
    # Simulate macOS: os has no sched_setaffinity
    import os as _os
    with mock.patch.object(_os, "cpu_count", return_value=4):
        had = hasattr(_os, "sched_setaffinity")
        if had:
            saved = _os.sched_setaffinity
            del _os.sched_setaffinity
        try:
            assert compute_split(True) == (None, None)
        finally:
            if had:
                _os.sched_setaffinity = saved


def test_pin_current_thread_none_is_noop():
    pin_current_thread(None)  # must not raise


def test_pin_current_thread_swallows_errors():
    with mock.patch("os.sched_setaffinity", side_effect=OSError, create=True):
        pin_current_thread({0})  # error swallowed, never fatal
