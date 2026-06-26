"""Tests for DrmFramebuffer pure logic — ioctl number computation + struct sizes.

The actual DRM ioctl path can only be exercised on real Linux with
`/dev/dri/card0` present, so those tests live in `docs/deployment-safety.md`
as manual validation. Here we verify the math that's known to be a common
source of bugs in ctypes-based DRM clients.
"""
from ctypes import sizeof

from pi_fpv_companion.video.drm_framebuffer import (
    _DRM_IOCTL_MODE_CREATE_DUMB,
    _DRM_IOCTL_MODE_GETCONNECTOR,
    _DRM_IOCTL_MODE_GETRESOURCES,
    _DRM_IOCTL_MODE_MAP_DUMB,
    _DRM_IOCTL_MODE_SETCRTC,
    _DrmModeCardRes,
    _DrmModeCreateDumb,
    _DrmModeCrtc,
    _DrmModeGetConnector,
    _DrmModeInfo,
    _DrmModeMapDumb,
    _iowr,
)


def test_create_dumb_struct_size():
    # Per drm/drm_mode.h: 6 u32 + 1 u64 = 24 + 8 = 32 bytes
    # The u64 lands on offset 24 which is naturally 8-aligned, so no padding.
    assert sizeof(_DrmModeCreateDumb) == 32


def test_map_dumb_struct_size():
    # u32 handle + u32 pad + u64 offset = 16
    assert sizeof(_DrmModeMapDumb) == 16


def test_modeinfo_struct_size():
    # Per drm_mode.h drm_mode_modeinfo: 68 bytes total
    assert sizeof(_DrmModeInfo) == 68


def test_card_res_struct_size():
    # 4 u64 + 8 u32 = 32 + 32 = 64
    assert sizeof(_DrmModeCardRes) == 64


def test_get_connector_struct_size():
    # 4 u64 + 12 u32 = 32 + 48 = 80
    assert sizeof(_DrmModeGetConnector) == 80


def test_crtc_struct_size():
    # u64 + 7 u32 + _DrmModeInfo(68) = 8 + 28 + 68 = 104
    assert sizeof(_DrmModeCrtc) == 104


def test_iowr_encodes_known_ioctl_numbers():
    # Spot-check against well-known DRM ioctl values from drm/drm_mode.h
    # DRM_IOCTL_MODE_CREATE_DUMB = 0xC02064B2 on a 64-bit system
    # (R/W direction, type 'd' = 0x64, size 0x28 = 40, NR 0xB2)
    val = _iowr(_DRM_IOCTL_MODE_CREATE_DUMB, _DrmModeCreateDumb)
    assert val == 0xC02064B2

    # DRM_IOCTL_MODE_MAP_DUMB = 0xC01064B3
    val = _iowr(_DRM_IOCTL_MODE_MAP_DUMB, _DrmModeMapDumb)
    assert val == 0xC01064B3


def test_constants_match_kernel_drm_mode_h():
    # Sanity check that we didn't typo any of the ioctl NR values
    assert _DRM_IOCTL_MODE_GETRESOURCES == 0xA0
    assert _DRM_IOCTL_MODE_SETCRTC == 0xA2
    assert _DRM_IOCTL_MODE_GETCONNECTOR == 0xA7
    assert _DRM_IOCTL_MODE_CREATE_DUMB == 0xB2
    assert _DRM_IOCTL_MODE_MAP_DUMB == 0xB3
