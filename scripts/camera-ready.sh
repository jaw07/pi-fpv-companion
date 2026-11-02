#!/usr/bin/env bash
# ExecStartPre gate: ensure the IMX500 is enumerable before the service launches.
#
# Recovery ladder, in order of cost:
#   rung 1  frame-watchdog os._exit() on a stall/no-first-frame  (in the Python process)
#   rung 2  systemd restarts the process -> reopens the camera in software (~3-5s)
#   rung 3  THIS script: wait for the camera to enumerate. The driver-level reset
#           (reset-camera.sh) is OPT-IN via CAMERA_RESET=1: its sysfs unbind has crashed
#           in the kernel driver, which is a plausible cause of hard resets.
#   rung 4  systemd start limit: stop restarting (no reboot)
#
# A plain capture hang leaves the camera ENUMERABLE, so the first poll passes and this
# exits immediately (rung 2 already fixed it). Only a sensor that fell off the bus makes
# us wait + reset. Always exits 0: start anyway and let rungs 1/4 handle a hard fault.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
# timeout: a wedged libcamera can hang rpicam-hello forever and stall the unit start.
listed() { timeout 5 rpicam-hello --list-cameras 2>/dev/null | grep -qi imx500; }
wait_for() { for _ in $(seq 1 "$1"); do listed && return 0; sleep 1; done; return 1; }

if wait_for 15; then exit 0; fi

if ! command -v rpicam-hello >/dev/null; then
  echo "camera-ready: rpicam-hello not installed (apt install rpicam-apps-lite) — skipping check" >&2
  exit 0
fi

if [ "${CAMERA_RESET:-0}" != "1" ]; then
  echo "camera-ready: imx500 not enumerable after 15s — starting anyway (CAMERA_RESET=1 to try a driver reset)" >&2
  exit 0
fi

echo "camera-ready: imx500 not enumerable after 15s — resetting camera subsystem" >&2
timeout 20 bash "$HERE/reset-camera.sh" || true

if wait_for 15; then exit 0; fi

echo "camera-ready: imx500 still not listed after reset — starting anyway" >&2
exit 0
