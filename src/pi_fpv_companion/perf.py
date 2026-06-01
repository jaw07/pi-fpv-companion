"""Per-tick latency and memory instrumentation.

The Mac is fast; the Pi Zero 2W is not. We always develop and test against an
explicit budget so a feature that works on Mac but blows up the Pi gets caught
before the hardware lands.

Detection runs on the IMX500's on-sensor NPU (~0 host ms), so the host budget is
just the pipeline scaffold (camera read + tracker + guidance + MAVLink), which is
cheap. Scaling — workload-dependent, MEASURED on a real Pi Zero 2W A53 1GHz
(against an M-series Mac running the same code):

    Pipeline scaffold (synth + IoU + MAVLink):  Mac 0.18 ms → Pi 0.40 ms  =  2.2×

The default `pi_scale_factor=6.0` is conservative for the host pipeline (real is
~2-3×) — or, better, run the profile script on the actual Pi.

Pi Zero 2W resource ceiling:
    RAM       512 MB total, ~350 MB usable after Bookworm Lite + libcamera
    CPU       quad-A53 @ 1 GHz, throttles under sustained load
    Tick      33 ms for 30 FPS, 50 ms for 20 FPS, 100 ms for 10 FPS
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass(frozen=True)
class PiBudget:
    """Hard ceilings to validate against. Tick budget drives the target frame rate."""
    max_tick_ms: float = 33.0          # 30 FPS
    max_rss_mb: float = 200.0          # leave headroom under Pi's ~350 MB usable
    pi_scale_factor: float = 6.0       # rough Mac→Pi multiplier


@dataclass
class PerfStats:
    n_ticks: int
    tick_p50_ms: float
    tick_p95_ms: float
    tick_p99_ms: float
    tick_max_ms: float
    rss_peak_mb: float
    rss_current_mb: float
    budget: PiBudget


class PerfMonitor:
    """Tracks per-tick wall time and peak RSS over a rolling window."""

    def __init__(self, budget: Optional[PiBudget] = None, window: int = 600) -> None:
        self._budget = budget or PiBudget()
        self._ticks_ms: Deque[float] = deque(maxlen=window)
        self._rss_peak_mb: float = 0.0
        self._proc = None  # lazy psutil.Process

    def tick_start(self) -> float:
        return time.perf_counter()

    def tick_end(self, started_at: float) -> float:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self._ticks_ms.append(elapsed_ms)
        self._update_rss()
        return elapsed_ms

    def _update_rss(self) -> None:
        if self._proc is None:
            import psutil
            self._proc = psutil.Process()
        rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        if rss_mb > self._rss_peak_mb:
            self._rss_peak_mb = rss_mb

    def stats(self) -> PerfStats:
        if not self._ticks_ms:
            return PerfStats(0, 0.0, 0.0, 0.0, 0.0, self._rss_peak_mb, 0.0, self._budget)
        ts = sorted(self._ticks_ms)
        n = len(ts)
        rss_now = 0.0
        if self._proc is not None:
            rss_now = self._proc.memory_info().rss / (1024 * 1024)
        return PerfStats(
            n_ticks=n,
            tick_p50_ms=ts[n // 2],
            tick_p95_ms=ts[min(n - 1, int(n * 0.95))],
            tick_p99_ms=ts[min(n - 1, int(n * 0.99))],
            tick_max_ms=ts[-1],
            rss_peak_mb=self._rss_peak_mb,
            rss_current_mb=rss_now,
            budget=self._budget,
        )

    def report(self) -> str:
        s = self.stats()
        b = s.budget
        scale = b.pi_scale_factor
        lines = [
            "Performance — Mac measured, Pi Zero 2W estimated:",
            f"  ticks       {s.n_ticks}",
            f"  p50 tick    {s.tick_p50_ms:6.2f} ms   (Pi est ~{s.tick_p50_ms * scale:6.1f} ms)",
            f"  p95 tick    {s.tick_p95_ms:6.2f} ms   (Pi est ~{s.tick_p95_ms * scale:6.1f} ms)",
            f"  p99 tick    {s.tick_p99_ms:6.2f} ms   (Pi est ~{s.tick_p99_ms * scale:6.1f} ms)",
            f"  max tick    {s.tick_max_ms:6.2f} ms",
            f"  peak RSS    {s.rss_peak_mb:6.1f} MB   (Pi budget {b.max_rss_mb:.0f} MB)",
            f"  budget      tick {b.max_tick_ms:.0f} ms  rss {b.max_rss_mb:.0f} MB",
        ]
        pi_p95 = s.tick_p95_ms * scale
        if pi_p95 > b.max_tick_ms:
            lines.append(
                f"  VERDICT     OVER tick budget on Pi  ({pi_p95:.1f} > {b.max_tick_ms:.0f} ms)"
            )
        elif s.rss_peak_mb > b.max_rss_mb:
            lines.append(f"  VERDICT     OVER RSS budget ({s.rss_peak_mb:.0f} > {b.max_rss_mb:.0f} MB)")
        else:
            lines.append("  VERDICT     fits Pi budget")
        return "\n".join(lines)
