#!/usr/bin/env bash
# ExecStartPre gate: ensure the IMX500 is enumerable before the service launches.
#
# Recovery ladder, in order of cost:
#   rung 1  frame-watchdog os._exit() on a stall/no-first-frame  (in the Python process)
#   rung 2  systemd restarts the process -> reopens the camera in software (~3-5s)
#   rung 3  THIS script: if the camera isn't even enumerable, reset the camera subsystem
#           (driver unbind/rebind, no reboot) and wait again  <-- the middle rung
#   rung 4  StartLimitAction=reboot if rung 3 keeps failing (full Pi reboot, last resort)
#
# A plain capture hang leaves the camera ENUMERABLE, so the first poll passes and this
# exits immediately (rung 2 already fixed it). Only a sensor that fell off the bus makes
# us wait + reset. Always exits 0: start anyway and let rungs 1/4 handle a hard fault.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
listed() { rpicam-hello --list-cameras 2>/dev/null | grep -qi imx500; }
wait_for() { for _ in $(seq 1 "$1"); do listed && return 0; sleep 1; done; return 1; }

if wait_for 15; then exit 0; fi

echo "camera-ready: imx500 not enumerable after 15s — resetting camera subsystem" >&2
bash "$HERE/reset-camera.sh" || true

if wait_for 15; then exit 0; fi

echo "camera-ready: imx500 still not listed after reset — starting anyway" >&2
exit 0
