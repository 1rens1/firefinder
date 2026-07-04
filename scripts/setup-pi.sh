#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip python3-picamera2 python3-opencv \
  imx500-all curl ca-certificates

UV_CMD=uv
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user uv
  UV_CMD="python3 -m uv"
fi

# Ensure a fresh environment that can access apt-installed system packages.
if [[ ! -f .venv/pyvenv.cfg ]] || ! grep -q '^include-system-site-packages = true$' .venv/pyvenv.cfg; then
  $UV_CMD venv --python /usr/bin/python3 --no-managed-python --system-site-packages --clear .venv
fi

$UV_CMD pip install --python .venv -e .

.venv/bin/python3 -c "import cv2; from picamera2 import Picamera2; from picamera2.devices import IMX500"