"""Companion flight recorder — the companion-side blackbox.

The FC's dataflash records what the FC *did*; nothing records what the companion
*saw and decided*. After flight 2 the diagnosis had to be reconstructed from code
because every per-tick decision (switch reading, target quality, gate verdict,
intent sent) was ephemeral. This module writes that decision trail as JSONL to a
rotating file under var/flight/ (the systemd unit's one writable path), at a
sampled rate so a 100-hour season costs tens of MB, not GB.

One line ≈ 200 bytes at 10 Hz ≈ 7 MB per 10 flight-hours. Files are opened
line-buffered (fflush after each newline), and fsynced to the SD card every
~1 second so a battery pull loses at most that window.
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


class FlightRecorder:
    """Rate-limited JSONL recorder for the pipeline's per-tick status. One file per
    process start (named by wall-clock start time), rotated by size, oldest pruned."""

    def __init__(self, directory: str | Path, rate_hz: float = 10.0,
                 max_bytes: int = 20_000_000, keep_files: int = 10) -> None:
        self._dir = Path(directory)
        self._min_period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._max_bytes = max_bytes
        self._keep = keep_files
        self._fh = None
        self._bytes = 0
        self._last_write = 0.0
        self._last_sync = 0.0
        self._sync_interval = 1.0
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._open_new()
        except OSError as e:
            # Recording must never block flight: no writable dir -> recorder off.
            _log.warning("flight recorder disabled: cannot write %s (%s)", self._dir, e)
            self._fh = None

    def _open_new(self) -> None:
        name = time.strftime("flight-%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.jsonl"
        self._fh = open(self._dir / name, "w", buffering=1)   # line-buffered: ≤1 line lost on power cut
        self._bytes = 0
        self._prune()

    def _prune(self) -> None:
        files = sorted(self._dir.glob("flight-*.jsonl"))
        for old in files[:-self._keep] if len(files) > self._keep else []:
            try:
                old.unlink()
            except OSError:
                pass

    def record(self, target, intent, gated, switch, armed: bool) -> None:
        """Write one sampled record. Cheap no-op between samples and when disabled."""
        if self._fh is None:
            return
        now = time.monotonic()
        if now - self._last_write < self._min_period:
            return
        self._last_write = now
        rec = {
            "t": round(time.time(), 3),
            "mode": switch.mode.name,
            "pwm": switch.pwm_us,
            "armed": armed,
            "muted": gated.muted,
            "reason": gated.reason,
            "yaw_dps": round(intent.yaw_rate_dps, 2),
            "pitch_deg": round(intent.pitch_deg, 2),
            "roll_deg": round(intent.roll_deg, 2),
            "thrust": round(intent.thrust, 3),
        }
        if intent.vertical_rate_mps is not None:
            rec["vz"] = round(intent.vertical_rate_mps, 2)
        if getattr(intent, "phase", None) is not None:
            rec["phase"] = intent.phase
        if getattr(intent, "pitch_rate_rads", None) is not None:
            rec["pr_rads"] = round(intent.pitch_rate_rads, 4)
        if getattr(intent, "roll_rate_rads", None) is not None:
            rec["rr_rads"] = round(intent.roll_rate_rads, 4)
        if getattr(intent, "yaw_rate_rads", None) is not None:
            rec["yr_rads"] = round(intent.yaw_rate_rads, 4)
        if getattr(intent, "hover", None) is not None:
            rec["hover"] = round(intent.hover, 3)
        if getattr(intent, "agl_m", None) is not None:
            rec["agl"] = round(intent.agl_m, 1)
        if target is not None:
            d = target.detection
            rec["tgt"] = {"x": d.x, "y": d.y, "w": d.w, "h": d.h,
                          "q": round(target.quality, 2), "id": target.track_id}
        try:
            line = json.dumps(rec, separators=(",", ":")) + "\n"
            self._fh.write(line)
            self._bytes += len(line)
            if now - self._last_sync >= self._sync_interval:
                os.fsync(self._fh.fileno())
                self._last_sync = now
            if self._bytes >= self._max_bytes:
                self._fh.close()
                self._open_new()
        except (OSError, ValueError) as e:
            _log.warning("flight recorder write failed (%s) — recorder off", e)
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
