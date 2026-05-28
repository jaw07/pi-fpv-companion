#!/usr/bin/env bash
# One-shot venv bootstrap. Handles the ncnn -> opencv-python -> contrib conflict
# by force-reinstalling opencv-contrib-python last (so its cv2 module wins on import).
#
# Run from project root:
#   bash scripts/setup-venv.sh
set -euo pipefail

PY="${PY:-python3}"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "creating venv at $VENV_DIR with $PY ..."
    # --system-site-packages so apt-installed python3-picamera2 + python3-libcamera
    # are importable from the venv (they're not pip-installable).
    "$PY" -m venv --system-site-packages "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"

echo "installing package + deps ..."
"$PIP" install -q -e .

# ncnn declares opencv-python as a dep. That installs alongside opencv-contrib-python
# and silently wins on import, breaking cv2.legacy.TrackerKCF/CSRT/etc. Force-reinstall
# the contrib package as the last step so its cv2.so is the one that ends up resolved.
echo "fixing opencv-python/contrib conflict ..."
"$PIP" uninstall -q -y opencv-python || true
"$PIP" install -q --force-reinstall --no-deps opencv-contrib-python

echo "installing dev deps ..."
"$PIP" install -q pytest

echo
echo "verifying:"
"$VENV_DIR/bin/python" -c "
import cv2
assert hasattr(cv2.legacy, 'TrackerKCF_create'), 'cv2.legacy.TrackerKCF_create missing — contrib package was overridden'
print(f'  cv2 version       {cv2.__version__}')
print(f'  cv2.legacy.KCF    ok')
"

echo
echo "setup complete. run tests with:"
echo "  $VENV_DIR/bin/pytest tests/"
