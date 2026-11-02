#!/usr/bin/env bash
# One-shot venv bootstrap. Ensures opencv-contrib-python wins on import (the classical
# cv2.legacy trackers need it) by force-reinstalling it last, in case any dependency
# pulls in plain opencv-python alongside it.
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

# Pin numpy to the version apt installed. The venv sees system packages (picamera2,
# simplejpeg) that were compiled against it; a pip-installed numpy 2.x would shadow it
# and crash them at import ("numpy.dtype size changed"). opencv is capped <4.12 in
# pyproject.toml because newer wheels require numpy>=2.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
SYS_NUMPY="$(/usr/bin/python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true)"
if [ -n "$SYS_NUMPY" ]; then
    echo "pinning numpy==$SYS_NUMPY (system)"
    echo "numpy==$SYS_NUMPY" > "$CONSTRAINTS"
fi

echo "installing package + deps ..."
"$PIP" install -q -c "$CONSTRAINTS" -e .

# If any dep pulls in opencv-python, it installs alongside opencv-contrib-python and can
# silently win on import, breaking cv2.legacy.TrackerKCF/CSRT/etc. Force-reinstall the
# contrib package as the last step so its cv2.so is the one that ends up resolved.
echo "ensuring opencv-contrib wins on import ..."
"$PIP" uninstall -q -y opencv-python || true
"$PIP" install -q --force-reinstall --no-deps "opencv-contrib-python>=4.8,<4.12"

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
