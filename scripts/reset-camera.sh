#!/usr/bin/env bash
# Best-effort reset of the IMX500 camera subsystem WITHOUT a full reboot — the middle
# rung of the camera recovery ladder (between a process restart and StartLimitAction=
# reboot). Targets the failure a process restart can't fix: the sensor dropping off the
# CSI/i2c bus (enumeration loss), as opposed to a libcamera capture hang (which a plain
# reopen already clears).
#
# Re-initialises by unbinding+rebinding the CSI receiver (platform) driver via sysfs, falling back to a kernel-module reload. Must run as root.
# Exit 0 if the camera enumerates afterwards, 1 if not (caller starts the service anyway).
set -u

log() { echo "reset-camera: $*" >&2; }
listed() { timeout 5 rpicam-hello --list-cameras 2>/dev/null | grep -qi imx500; }

# (The i2c sensor unbind/rebind step was removed: the unbind crashed inside the imx500
#  kernel driver, which can take the whole Pi down.)

# 1) Unbind/rebind the CSI receiver (unicam / rp1-cfe) platform driver — heavier reset.
for drv in /sys/bus/platform/drivers/unicam \
           /sys/bus/platform/drivers/bcm2835-unicam \
           /sys/bus/platform/drivers/rp1-cfe; do
  [ -d "$drv" ] || continue
  for dev in "$drv"/*; do
    [ -L "$dev" ] || continue          # only device symlinks, not bind/unbind/uevent
    bn="$(basename "$dev")"
    log "platform unbind/rebind $bn ($(basename "$drv"))"
    echo "$bn" > "$drv/unbind" 2>/dev/null || true
    sleep 1
    echo "$bn" > "$drv/bind" 2>/dev/null || true
  done
done
sleep 2
listed && { log "camera back after CSI rebind"; exit 0; }

# 2) Fallback: reload the sensor kernel module (no-op if built-in).
log "still absent — trying imx500 module reload"
modprobe -r imx500 2>/dev/null || true
sleep 1
modprobe imx500 2>/dev/null || true
sleep 2

if listed; then
  log "camera back after module reload"
  exit 0
fi
log "camera NOT recovered by subsystem reset — caller should escalate"
exit 1
