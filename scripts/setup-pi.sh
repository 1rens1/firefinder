#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

sudo apt-get install -y imx500-all

if [[ ! -f .venv/pyvenv.cfg ]] || ! grep -q '^include-system-site-packages = true$' .venv/pyvenv.cfg; then
  uv venv --python /usr/bin/python3 --system-site-packages --clear .venv
fi

uv sync
uv run python -c "import cv2; from picamera2 import Picamera2; from picamera2.devices import IMX500"
