"""DRM dumb-buffer framebuffer — Trixie+ replacement for `/dev/fb0`.

Modern Pi OS (Bookworm with default KMS, Trixie always) no longer auto-creates
`/dev/fb0`. The only path to drive the composite output's pixels is to open
`/dev/dri/card0`, find the Composite connector, allocate a "dumb" buffer that
matches the connector's mode, mmap it, and bind it as the active framebuffer.

This is a thin ctypes wrapper around just the half-dozen DRM ioctls we need —
no libdrm Python binding required.

Implementation notes:
  - Format: XR24 (= BGRA 8888, alpha ignored). Most A53 KMS drivers support
    this. RGB565 is also supported by vc4 but we go with 32bpp for cleaner
    BGR→buffer conversion (same layout as our overlay code).
  - Single-buffered, no page-flip. We write in-place. Tearing is invisible
    over composite analog; double-buffer can be added later if needed.
  - We save the connector's existing CRTC mode on open and restore it on
    close, so fbcon (the kernel framebuffer console) gets the display back
    when our process exits.

Run as root (or in `video` group) to be allowed to SET_MASTER and SETCRTC.
"""
from __future__ import annotations
import ctypes
import fcntl
import mmap
import os
from ctypes import c_char, c_uint16, c_uint32, c_uint64, sizeof
from typing import Optional

import numpy as np


# ---- ioctl number computation (mirrors include/asm-generic/ioctl.h) ----

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

_DRM_TYPE = ord("d")


def _ioc(dir_: int, nr: int, size: int) -> int:
    return (
        (dir_ << _IOC_DIRSHIFT)
        | (_DRM_TYPE << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _iowr(nr: int, struct_cls) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, nr, sizeof(struct_cls))


# ---- DRM constants (from drm/drm.h and drm/drm_mode.h) ----

_DRM_IOCTL_MODE_GETRESOURCES = 0xA0
_DRM_IOCTL_MODE_GETCRTC      = 0xA1
_DRM_IOCTL_MODE_SETCRTC      = 0xA2
_DRM_IOCTL_MODE_GETENCODER   = 0xA6
_DRM_IOCTL_MODE_GETCONNECTOR = 0xA7
_DRM_IOCTL_MODE_ADDFB        = 0xAE
_DRM_IOCTL_MODE_RMFB         = 0xAF
_DRM_IOCTL_MODE_CREATE_DUMB  = 0xB2
_DRM_IOCTL_MODE_MAP_DUMB     = 0xB3
_DRM_IOCTL_MODE_DESTROY_DUMB = 0xB4

_DRM_MODE_CONNECTOR_Composite = 5
_DRM_MODE_CONNECTOR_TV        = 6   # some kernels report TV instead of Composite

_DRM_MODE_CONNECTED = 1
_DRM_MODE_DISCONNECTED = 2
_DRM_MODE_UNKNOWN = 3      # composite has no EDID — kernel reports "unknown" by default


# ---- ctypes struct definitions (only the fields we use) ----


class _DrmModeCardRes(ctypes.Structure):
    _fields_ = [
        ("fb_id_ptr",        c_uint64),
        ("crtc_id_ptr",      c_uint64),
        ("connector_id_ptr", c_uint64),
        ("encoder_id_ptr",   c_uint64),
        ("count_fbs",        c_uint32),
        ("count_crtcs",      c_uint32),
        ("count_connectors", c_uint32),
        ("count_encoders",   c_uint32),
        ("min_width",        c_uint32),
        ("max_width",        c_uint32),
        ("min_height",       c_uint32),
        ("max_height",       c_uint32),
    ]


class _DrmModeInfo(ctypes.Structure):
    _fields_ = [
        ("clock",       c_uint32),
        ("hdisplay",    c_uint16),
        ("hsync_start", c_uint16),
        ("hsync_end",   c_uint16),
        ("htotal",      c_uint16),
        ("hskew",       c_uint16),
        ("vdisplay",    c_uint16),
        ("vsync_start", c_uint16),
        ("vsync_end",   c_uint16),
        ("vtotal",      c_uint16),
        ("vscan",       c_uint16),
        ("vrefresh",    c_uint32),
        ("flags",       c_uint32),
        ("type",        c_uint32),
        ("name",        c_char * 32),
    ]


class _DrmModeGetConnector(ctypes.Structure):
    _fields_ = [
        ("encoders_ptr",       c_uint64),
        ("modes_ptr",          c_uint64),
        ("props_ptr",          c_uint64),
        ("prop_values_ptr",    c_uint64),
        ("count_modes",        c_uint32),
        ("count_props",        c_uint32),
        ("count_encoders",     c_uint32),
        ("encoder_id",         c_uint32),
        ("connector_id",       c_uint32),
        ("connector_type",     c_uint32),
        ("connector_type_id",  c_uint32),
        ("connection",         c_uint32),
        ("mm_width",           c_uint32),
        ("mm_height",          c_uint32),
        ("subpixel",           c_uint32),
        ("pad",                c_uint32),
    ]


class _DrmModeGetEncoder(ctypes.Structure):
    _fields_ = [
        ("encoder_id",      c_uint32),
        ("encoder_type",    c_uint32),
        ("crtc_id",         c_uint32),
        ("possible_crtcs",  c_uint32),
        ("possible_clones", c_uint32),
    ]


class _DrmModeCreateDumb(ctypes.Structure):
    _fields_ = [
        ("height", c_uint32),
        ("width",  c_uint32),
        ("bpp",    c_uint32),
        ("flags",  c_uint32),
        ("handle", c_uint32),
        ("pitch",  c_uint32),
        ("size",   c_uint64),
    ]


class _DrmModeMapDumb(ctypes.Structure):
    _fields_ = [
        ("handle", c_uint32),
        ("pad",    c_uint32),
        ("offset", c_uint64),
    ]


class _DrmModeDestroyDumb(ctypes.Structure):
    _fields_ = [("handle", c_uint32)]


class _DrmModeFbCmd(ctypes.Structure):
    _fields_ = [
        ("fb_id",  c_uint32),
        ("width",  c_uint32),
        ("height", c_uint32),
        ("pitch",  c_uint32),
        ("bpp",    c_uint32),
        ("depth",  c_uint32),
        ("handle", c_uint32),
    ]


class _DrmModeCrtc(ctypes.Structure):
    _fields_ = [
        ("set_connectors_ptr", c_uint64),
        ("count_connectors",   c_uint32),
        ("crtc_id",            c_uint32),
        ("fb_id",              c_uint32),
        ("x",                  c_uint32),
        ("y",                  c_uint32),
        ("gamma_size",         c_uint32),
        ("mode_valid",         c_uint32),
        ("mode",               _DrmModeInfo),
    ]


# ---- the actual framebuffer class ----


class DrmFramebuffer:
    """Framebuffer Protocol implementation backed by a DRM dumb buffer."""

    def __init__(self, device: str = "/dev/dri/card0") -> None:
        self._device = device
        self._fd: Optional[int] = None
        self._handle: int = 0
        self._fb_id: int = 0
        self._crtc_id: int = 0
        self._connector_id: int = 0
        self._size: int = 0
        self._mmap: Optional[mmap.mmap] = None
        self._buf_view: Optional[np.ndarray] = None    # numpy view of the mmap as BGRA
        self._saved_crtc: Optional[_DrmModeCrtc] = None
        self.width: int = 0
        self.height: int = 0
        self.bpp: int = 32                              # XR24 / BGRA
        self._stride: int = 0

    # The Framebuffer Protocol's `open` does all the heavy DRM lifting.
    def open(self) -> None:
        self._fd = os.open(self._device, os.O_RDWR | os.O_CLOEXEC)

        connector, mode = self._find_composite_connector()
        if connector is None or mode is None:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(
                "No connected Composite/TV connector found on /dev/dri/card0. "
                "Check /boot/firmware/config.txt has `dtoverlay=vc4-kms-v3d,composite` "
                "and /boot/firmware/cmdline.txt has `vc4.tv_norm=PAL` (or NTSC)."
            )

        # Any failure past this point must release fd + dumb buffer + mmap,
        # not leak them. close() is safe on partial state (per-resource guards).
        try:
            self._connector_id = connector.connector_id
            self._crtc_id = self._find_crtc_for_encoder(connector.encoder_id)

            # Snapshot the existing CRTC mode so we can restore fbcon on close
            self._saved_crtc = _DrmModeCrtc(crtc_id=self._crtc_id)
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETCRTC, _DrmModeCrtc), self._saved_crtc)

            # Allocate a dumb buffer matching the mode (XR24 / BGRA, 32 bpp)
            create = _DrmModeCreateDumb(
                width=mode.hdisplay, height=mode.vdisplay, bpp=32, flags=0
            )
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_CREATE_DUMB, _DrmModeCreateDumb), create)
            self._handle = create.handle
            self._stride = create.pitch
            self._size = create.size
            self.width = create.width
            self.height = create.height

            # Add it as an FB object (depth=24 for XR24)
            addfb = _DrmModeFbCmd(
                width=mode.hdisplay, height=mode.vdisplay,
                pitch=self._stride, bpp=32, depth=24, handle=self._handle,
            )
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_ADDFB, _DrmModeFbCmd), addfb)
            self._fb_id = addfb.fb_id

            # Map it
            mapdumb = _DrmModeMapDumb(handle=self._handle)
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_MAP_DUMB, _DrmModeMapDumb), mapdumb)
            self._mmap = mmap.mmap(
                self._fd, self._size,
                mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                offset=mapdumb.offset,
            )

            # Numpy view over the mapped memory as HxWx4 BGRA — write via this for speed
            view = np.frombuffer(self._mmap, dtype=np.uint8, count=self._size)
            # The buffer may have row padding (stride > width*4); reshape as
            # (height, stride/4, 4) then slice to (height, width, 4) when writing.
            self._buf_view = view.reshape((self.height, self._stride // 4, 4))[:, : self.width]

            # Activate: bind our FB to the CRTC + connector, with this mode
            set_crtc = _DrmModeCrtc(
                crtc_id=self._crtc_id,
                fb_id=self._fb_id,
                x=0, y=0,
                mode_valid=1,
                mode=mode,
                count_connectors=1,
            )
            conn_arr = (c_uint32 * 1)(self._connector_id)
            set_crtc.set_connectors_ptr = ctypes.addressof(conn_arr)
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_SETCRTC, _DrmModeCrtc), set_crtc)
        except Exception:
            self.close()
            raise

    def _find_composite_connector(self):
        assert self._fd is not None
        res = _DrmModeCardRes()
        fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETRESOURCES, _DrmModeCardRes), res)
        # Allocate arrays sized to the counts the kernel just reported, then ioctl again
        # with pointers so the kernel fills them in.
        conns = (c_uint32 * res.count_connectors)()
        crtcs = (c_uint32 * res.count_crtcs)()
        encoders = (c_uint32 * res.count_encoders)()
        fbs = (c_uint32 * res.count_fbs)()
        res.connector_id_ptr = ctypes.addressof(conns)
        res.crtc_id_ptr = ctypes.addressof(crtcs)
        res.encoder_id_ptr = ctypes.addressof(encoders)
        res.fb_id_ptr = ctypes.addressof(fbs)
        fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETRESOURCES, _DrmModeCardRes), res)

        for cid in conns:
            conn = _DrmModeGetConnector(connector_id=cid)
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETCONNECTOR, _DrmModeGetConnector), conn)
            if conn.connector_type not in (
                _DRM_MODE_CONNECTOR_Composite, _DRM_MODE_CONNECTOR_TV,
            ):
                continue
            # Composite has no EDID so the kernel reports state=unknown; treat it
            # the same as connected. Skip only the explicit "disconnected" case
            # (which would mean the SoC isn't driving the pad at all).
            if conn.connection == _DRM_MODE_DISCONNECTED:
                continue
            # Re-fetch with allocated arrays
            modes = (_DrmModeInfo * conn.count_modes)()
            encoders = (c_uint32 * conn.count_encoders)()
            props = (c_uint32 * conn.count_props)()
            prop_values = (c_uint64 * conn.count_props)()
            conn.modes_ptr = ctypes.addressof(modes)
            conn.encoders_ptr = ctypes.addressof(encoders)
            conn.props_ptr = ctypes.addressof(props)
            conn.prop_values_ptr = ctypes.addressof(prop_values)
            fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETCONNECTOR, _DrmModeGetConnector), conn)
            if conn.count_modes == 0:
                continue
            return conn, modes[0]  # preferred mode
        return None, None

    def _find_crtc_for_encoder(self, encoder_id: int) -> int:
        assert self._fd is not None
        enc = _DrmModeGetEncoder(encoder_id=encoder_id)
        fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETENCODER, _DrmModeGetEncoder), enc)
        if enc.crtc_id != 0:
            return enc.crtc_id
        # Walk possible_crtcs bitmap if encoder isn't currently bound
        res = _DrmModeCardRes()
        fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETRESOURCES, _DrmModeCardRes), res)
        crtcs = (c_uint32 * res.count_crtcs)()
        res.crtc_id_ptr = ctypes.addressof(crtcs)
        # Re-fetch with array
        res.count_connectors = 0
        res.count_encoders = 0
        res.count_fbs = 0
        fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_GETRESOURCES, _DrmModeCardRes), res)
        for i, cid in enumerate(crtcs):
            if enc.possible_crtcs & (1 << i):
                return cid
        raise RuntimeError("no usable CRTC found for the encoder")

    def write(self, bgr: np.ndarray) -> None:
        """Write a BGR uint8 frame. Resizes/clamps to the buffer size mismatch — caller
        is expected to deliver an image matching width × height (we validate)."""
        if self._buf_view is None:
            raise RuntimeError("framebuffer not open")
        h, w = bgr.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(
                f"frame {w}x{h} does not match DRM mode {self.width}x{self.height}"
            )
        # BGR -> BGRA: write B/G/R into channels 0/1/2, leave alpha as-is (any)
        self._buf_view[..., 0:3] = bgr

    def close(self) -> None:
        if self._fd is None:
            return
        # Best-effort restore of the original CRTC mode for fbcon
        try:
            if self._saved_crtc is not None:
                empty = (c_uint32 * 0)()
                self._saved_crtc.set_connectors_ptr = ctypes.addressof(empty)
                fcntl.ioctl(self._fd, _iowr(_DRM_IOCTL_MODE_SETCRTC, _DrmModeCrtc), self._saved_crtc)
        except OSError:
            pass
        # Drop the numpy view first — it holds a reference to the mmap and would
        # otherwise raise "cannot close exported pointers exist" on close().
        self._buf_view = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        # Detach FB, destroy dumb
        try:
            if self._fb_id:
                rmfb = c_uint32(self._fb_id)
                fcntl.ioctl(self._fd, _ioc(_IOC_WRITE | _IOC_READ, _DRM_IOCTL_MODE_RMFB, sizeof(rmfb)), rmfb)
        except OSError:
            pass
        try:
            if self._handle:
                destroy = _DrmModeDestroyDumb(handle=self._handle)
                fcntl.ioctl(
                    self._fd,
                    _iowr(_DRM_IOCTL_MODE_DESTROY_DUMB, _DrmModeDestroyDumb),
                    destroy,
                )
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None
