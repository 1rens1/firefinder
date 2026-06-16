#!/usr/bin/env bash

LOG="./pi-power-$(date '+%Y%m%d-%H%M%S').log"

echo "Logging to $LOG"
echo "Press Ctrl+C to stop"

while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  THROTTLED=$(vcgencmd get_throttled | cut -d= -f2)
  TEMP=$(vcgencmd measure_temp | cut -d= -f2)

  echo "$TS throttled=$THROTTLED temp=$TEMP" | tee -a "$LOG"

  sleep 0.5
done