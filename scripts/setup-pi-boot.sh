#!/usr/bin/env bash
# Pi-side boot configuration for pi-fpv-companion. Idempotent — safe to re-run.
#
# What this changes:
#   /boot/firmware/config.txt:
#     - dtoverlay=disable-bt           (free PL011 for ttyAMA0)
#     - dtoverlay=vc4-kms-v3d,composite (analog CVBS via DRM, Trixie-correct)
#   /boot/firmware/cmdline.txt:
#     - remove `console=serial0,<baud>` (don't let the kernel grab our UART)
#     - add    `vc4.tv_norm=PAL`        (PAL composite; pass --ntsc for NTSC)
#   systemd:
#     - disable serial-getty@ttyAMA0 and @ttyS0
#
# A reboot is required after running this for the changes to take effect.

set -euo pipefail

CONFIG="/boot/firmware/config.txt"
CMDLINE="/boot/firmware/cmdline.txt"
TV_NORM="PAL"

while [ $# -gt 0 ]; do
  case "$1" in
    --ntsc) TV_NORM="NTSC"; shift ;;
    --pal)  TV_NORM="PAL"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$EUID" -ne 0 ]; then
  echo "this script requires root (sudo bash scripts/setup-pi-boot.sh)" >&2
  exit 1
fi

# One-time backup before first edit
if [ ! -f "$CONFIG.bak" ]; then cp "$CONFIG"  "$CONFIG.bak"; fi
if [ ! -f "$CMDLINE.bak" ]; then cp "$CMDLINE" "$CMDLINE.bak"; fi

# --- config.txt ---
add_line_once() {
  local line="$1"
  grep -qxF "$line" "$CONFIG" || echo "$line" >> "$CONFIG"
}

# Strip legacy/ignored composite options if present (these are no-ops under
# default KMS but kept clean to avoid confusion)
sed -i -E '/^(enable_tvout|sdtv_mode|sdtv_aspect)=/d' "$CONFIG"

# Comment out a pre-existing PLAIN vc4-kms-v3d overlay: stock images ship one, and
# leaving it alongside the ,composite variant below loads KMS twice (a duplicate
# overlay that suppresses the composite connector). Replace it with the composite
# form rather than stacking. Idempotent — a line already commented is left alone.
sed -i -E 's/^dtoverlay=vc4-kms-v3d$/#&  # pi-fpv-companion: superseded by composite overlay below/' "$CONFIG"

echo >> "$CONFIG"
add_line_once "# pi-fpv-companion: free PL011 for ttyAMA0 (move BT to mini-UART)"
add_line_once "dtoverlay=disable-bt"
add_line_once "# pi-fpv-companion: composite output via vc4 KMS (Trixie path)"
add_line_once "dtoverlay=vc4-kms-v3d,composite"

# --- cmdline.txt ---
# Single-line file. Remove `console=serial0,<baud>` if present, then ensure
# vc4.tv_norm=<TV_NORM> is set (replace existing or append)
sed -i -E "s/(^| )console=serial0,[0-9]+( |$)/ /g; s/  +/ /g; s/^ +//; s/ +$//" "$CMDLINE"
if grep -q "vc4.tv_norm=" "$CMDLINE"; then
  sed -i -E "s/vc4.tv_norm=[A-Za-z]+/vc4.tv_norm=$TV_NORM/" "$CMDLINE"
else
  # Append (single-line file, no newline tail)
  sed -i "s|$| vc4.tv_norm=$TV_NORM|" "$CMDLINE"
fi

# --- serial getty ---
systemctl disable --now serial-getty@ttyAMA0.service 2>/dev/null || true
systemctl disable --now serial-getty@ttyS0.service   2>/dev/null || true

echo
echo "boot config applied (TV_NORM=$TV_NORM)."
echo "  config.txt tail:"
tail -8 "$CONFIG" | sed 's/^/    /'
echo
echo "  cmdline.txt:"
echo "    $(cat "$CMDLINE")"
echo
echo "REBOOT required for changes to take effect."
