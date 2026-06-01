#!/usr/bin/env bash
# Install pi-fpv-companion on a Raspberry Pi running Bookworm 64-bit Lite.
# Run on the Pi itself, as a user with sudo:
#   curl -fsSL <repo>/scripts/install-pi.sh | bash
#   # or:
#   bash scripts/install-pi.sh           # if you already cloned the repo
#
# What this does:
#   1. apt deps (python3-picamera2, libcamera, etc.)
#   2. sync repo to /opt/pi-fpv-companion
#   3. bootstrap .venv there, install package (-e) + fix opencv conflict
#   4. install systemd unit, enable but DON'T start (you set config first)

set -euo pipefail

INSTALL_DIR="/opt/pi-fpv-companion"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${USER:-pi}"

echo "==> pi-fpv-companion install"
echo "    source:      $SRC_DIR"
echo "    destination: $INSTALL_DIR"
echo "    service as:  $SERVICE_USER"
echo

# ---- apt deps ----
echo "==> apt deps"
sudo apt-get update
sudo apt-get install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-venv \
    python3-pip \
    git \
    rsync \
    libgl1 \
    libglib2.0-0t64

read -r -p "==> install IMX500 firmware + models (imx500-all)? [y/N] " yn
if [[ "$yn" =~ ^[Yy] ]]; then
    sudo apt-get install -y imx500-all
fi

# ---- sync repo ----
echo "==> syncing source to $INSTALL_DIR"
sudo mkdir -p "$INSTALL_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
rsync -av \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='/tmp' \
    "$SRC_DIR/" "$INSTALL_DIR/"

# Writable dir for runtime state (model downloads, log files, etc.)
mkdir -p "$INSTALL_DIR/var"

# ---- venv ----
echo "==> venv bootstrap"
cd "$INSTALL_DIR"
bash scripts/setup-venv.sh

# ---- systemd unit ----
echo "==> systemd unit"
sudo sed "s/^User=pi$/User=$SERVICE_USER/" systemd/pi-fpv-companion.service \
    | sudo tee /etc/systemd/system/pi-fpv-companion.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable pi-fpv-companion

# ---- UART / Bluetooth ----
read -r -p "==> apply boot config now (UART + composite via setup-pi-boot.sh)? [y/N] " yn
if [[ "$yn" =~ ^[Yy] ]]; then
    sudo bash "$INSTALL_DIR/scripts/setup-pi-boot.sh"
    echo "    boot config applied — reboot when ready."
else
    echo "    skipped. Run later with: sudo bash $INSTALL_DIR/scripts/setup-pi-boot.sh"
fi

echo
echo "==> install complete"
echo "    edit:  $INSTALL_DIR/config/imx500.yaml"
echo "    start: sudo systemctl start pi-fpv-companion"
echo "    logs:  journalctl -u pi-fpv-companion -f"
