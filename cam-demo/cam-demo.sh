#!/usr/bin/env bash
set -e

PORT=8080
IP=$(ip route get 1.1.1.1 | awk '{print $7; exit}')

command -v ffmpeg >/dev/null || {
  echo "ffmpeg missing: sudo apt update && sudo apt install -y ffmpeg"
  exit 1
}

command -v rpicam-vid >/dev/null || {
  echo "rpicam-vid missing"
  exit 1
}

echo "Open: http://${IP}:${PORT}"

while true; do
  rpicam-vid -n -t 0 --inline --codec mjpeg -o - | \
  ffmpeg -loglevel warning -f mjpeg -i - \
    -listen 1 -f mpjpeg "http://0.0.0.0:${PORT}"

  echo "Stream stopped. Restarting..."
  sleep 1
done