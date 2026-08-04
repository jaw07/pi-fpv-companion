"""Is the ~4-minute camera stall OUR software, or below us?

Runs a MINIMAL picamera2 capture loop — no Pipeline, no tracker, no FC, no
framebuffer, no threads — and reports any gap between frames longer than the
watchdog threshold. If this stalls too, nothing in the application can be
responsible.

Also answers time-driven vs frame-count-driven: pass --fps to halve the frame
rate. If the stall interval stays ~constant in SECONDS it is time-driven
(thermal / electrical); if it stays ~constant in FRAMES it is a counter or
handle leak below us, and halving the rate should roughly double the interval.

    .venv/bin/python var/hit/bare_capture_soak.py --minutes 12 --fps 22
    .venv/bin/python var/hit/bare_capture_soak.py --minutes 20 --fps 11
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--fps", type=float, default=22.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--stall-s", type=float, default=2.0,
                    help="gap that counts as a stall (matches the camera watchdog)")
    ap.add_argument("--hang-exit-s", type=float, default=15.0,
                    help="hard-exit if capture_request() blocks this long (0 = never). "
                         "See the watchdog note below — do not disable casually.")
    ap.add_argument("--model", default="/opt/pi-fpv-companion/models/"
                                       "imx500_network_yolo11n_416_pp.rpk")
    ap.add_argument("--no-model", action="store_true",
                    help="plain camera, no NPU firmware upload at all")
    args = ap.parse_args()

    from picamera2 import Picamera2

    imx500 = None
    if not args.no_model:
        from picamera2.devices import IMX500
        imx500 = IMX500(args.model)
        cam_num = imx500.camera_num
    else:
        cam_num = 0

    picam = Picamera2(cam_num)
    cfg = picam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "BGR888"},
        raw=None, controls={"FrameRate": float(args.fps)}, buffer_count=4)
    picam.configure(cfg)
    if imx500 is not None:
        imx500.show_network_fw_progress_bar()
    picam.start(cfg, show_preview=False)

    print(f"bare capture soak: {args.fps:g} fps, {args.minutes:g} min, "
          f"model={'none' if args.no_model else Path(args.model).name}", flush=True)

    # WATCHDOG — do not remove. After the CSI timeout, capture_request() NEVER returns;
    # it hangs in the driver forever. Without this the whole box can wedge, and systemd
    # runs the BCM2835 hardware watchdog at a 60s timeout, so a wedged box gets HARD
    # RESET. That is the suspected cause of the 2026-08-04 reboot during an earlier,
    # unguarded run of this very script. Production has the same protection in
    # Pipeline._start_camera_watchdog (os._exit ~2s after frames stop), which is very
    # likely why the rig survived a month of stalls without ever resetting.
    if args.hang_exit_s > 0:
        import os
        import threading
        state = {"last": time.monotonic()}

        def _watch():
            while True:
                time.sleep(1.0)
                stuck = time.monotonic() - state["last"]
                if stuck > args.hang_exit_s:
                    print(f"\nHUNG: capture_request() blocked {stuck:.0f}s "
                          f"(> --hang-exit-s {args.hang_exit_s:g}) — exiting before the "
                          f"box wedges and the hardware watchdog resets it.", flush=True)
                    os._exit(7)
        threading.Thread(target=_watch, daemon=True, name="hang-watchdog").start()
    else:
        state = {"last": time.monotonic()}

    t_end = time.monotonic() + args.minutes * 60.0
    t_start = time.monotonic()
    last = time.monotonic()
    n = 0
    stalls = []
    while time.monotonic() < t_end:
        req = picam.capture_request()
        req.release()
        now = time.monotonic()
        state["last"] = now                       # feed the hang watchdog
        gap = now - last
        last = now
        n += 1
        if gap > args.stall_s:
            stalls.append((now - t_start, n, gap))
            print(f"  STALL at t={now - t_start:7.1f}s  frame={n:6d}  gap={gap:5.2f}s",
                  flush=True)
    picam.stop()
    picam.close()

    elapsed = time.monotonic() - t_start
    print(f"\ndone: {elapsed:.0f}s, {n} frames ({n/elapsed:.1f} fps), "
          f"{len(stalls)} stall(s)", flush=True)
    if len(stalls) >= 2:
        secs = [b[0] - a[0] for a, b in zip(stalls, stalls[1:])]
        frames = [b[1] - a[1] for a, b in zip(stalls, stalls[1:])]
        print(f"  interval between stalls: {[f'{s:.0f}s' for s in secs]}")
        print(f"  frames between stalls:   {frames}")
    elif stalls:
        print(f"  first stall at {stalls[0][0]:.0f}s / frame {stalls[0][1]}")
    elif args.no_model:
        print("  NO STALLS with NO NPU model loaded. Compare against an identical run"
              "\n  WITH the model: if that one stalls, the fault is in the IMX500 network"
              "\n  path (firmware upload / NPU inference / its extra power draw), NOT the"
              "\n  CSI cable — a bad ribbon would not care whether a network is loaded.")
    else:
        print("  NO STALLS with the model loaded — the application above this loop is"
              "\n  implicated (this bare loop is everything except our Pipeline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
