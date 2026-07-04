#!/usr/bin/env python3

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = 8080
WIDTH = 32
HEIGHT = 24
PIXELS = WIDTH * HEIGHT
I2C_FREQUENCY = 800000
SENSOR_REFRESH_RATE = "REFRESH_4_HZ"
POLL_SECONDS = 0.0


class ThermalFrame:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.temperatures: list[float] | None = None
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.timestamp: float | None = None

    def update(self, frame: list[float]) -> None:
        minimum = min(frame)
        maximum = max(frame)

        with self.cond:
            self.temperatures = frame.copy()
            self.minimum = minimum
            self.maximum = maximum
            self.timestamp = time.time()
            self.cond.notify_all()

    def snapshot(self) -> dict[str, Any] | None:
        with self.cond:
            if self.temperatures is None:
                return None

            return {
                "width": WIDTH,
                "height": HEIGHT,
                "temperatures": self.temperatures,
                "min": self.minimum,
                "max": self.maximum,
                "timestamp": self.timestamp,
            }


thermal_frame = ThermalFrame()
startup_ready = threading.Event()
startup_error: str | None = None


INDEX_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Firefinder MLX90640</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #100818;
      color: #f8f2ff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background: #100818;
    }

    main { width: min(960px, 100%); }

    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.6rem, 3vw, 2.4rem);
      font-weight: 750;
    }

    #status {
      color: #cbb9da;
      font-size: 0.95rem;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .display {
      display: grid;
      gap: 14px;
    }

    canvas {
      width: 100%;
      aspect-ratio: 4 / 3;
      image-rendering: pixelated;
      border: 1px solid #3d2854;
      background: #1a0f24;
    }

    .range {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 12px;
      color: #f1e6fa;
      font-variant-numeric: tabular-nums;
    }

    .bar {
      height: 18px;
      border-radius: 999px;
      border: 1px solid #49305f;
      background: linear-gradient(90deg, #2a0b58, #6130a4, #b86dd5, #ffe85c);
    }

    @media (max-width: 620px) {
      body { padding: 14px; }
      header { display: block; }
      #status {
        margin-top: 4px;
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Firefinder MLX90640</h1>
      <div id="status">Waiting for thermal frame...</div>
    </header>

    <section class="display" aria-label="Thermal camera feed">
      <canvas id="thermal" width="32" height="24"></canvas>
      <div class="range" aria-label="Temperature range">
        <span id="min">--.- C</span>
        <div class="bar"></div>
        <span id="max">--.- C</span>
      </div>
    </section>
  </main>

  <script>
    const canvas = document.getElementById("thermal");
    const context = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const minEl = document.getElementById("min");
    const maxEl = document.getElementById("max");

    let lastTimestamp = 0;
    let lastBrowserTime = performance.now();
    let fps = 0;

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function mix(a, b, t) {
      return Math.round(a + (b - a) * t);
    }

    function thermalColor(t) {
      const stops = [
        [42, 11, 88],
        [97, 48, 164],
        [184, 109, 213],
        [255, 232, 92],
      ];

      const scaled = clamp(t, 0, 1) * (stops.length - 1);
      const index = Math.min(stops.length - 2, Math.floor(scaled));
      const local = scaled - index;
      const start = stops[index];
      const end = stops[index + 1];

      return [
        mix(start[0], end[0], local),
        mix(start[1], end[1], local),
        mix(start[2], end[2], local),
        255,
      ];
    }

    function draw(frame) {
      const image = context.createImageData(frame.width, frame.height);
      const min = frame.min;
      const max = frame.max;
      const span = Math.max(max - min, 0.1);

      frame.temperatures.forEach((temperature, pixel) => {
        const color = thermalColor((temperature - min) / span);
        const offset = pixel * 4;
        image.data[offset] = color[0];
        image.data[offset + 1] = color[1];
        image.data[offset + 2] = color[2];
        image.data[offset + 3] = color[3];
      });

      context.putImageData(image, 0, 0);
      minEl.textContent = `${min.toFixed(1)} C`;
      maxEl.textContent = `${max.toFixed(1)} C`;

      if (frame.timestamp !== lastTimestamp) {
        const now = performance.now();
        fps = 1000 / Math.max(now - lastBrowserTime, 1);
        lastBrowserTime = now;
        lastTimestamp = frame.timestamp;
      }

      statusEl.textContent =
        `Updated ${new Date(frame.timestamp * 1000).toLocaleTimeString()} | ${fps.toFixed(1)} FPS`;
    }

    async function poll() {
      try {
        const response = await fetch("/frame", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        draw(await response.json());
      } catch (error) {
        statusEl.textContent = "Waiting for thermal frame...";
      } finally {
        setTimeout(poll, 100);
      }
    }

    poll();
  </script>
</body>
</html>
"""


def write_json(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return

        if self.path == "/frame":
            frame = thermal_frame.snapshot()
            if frame is None:
                write_json(self, 503, {"error": "No thermal frame is ready yet"})
                return

            write_json(self, 200, frame)
            return

        self.send_response(404)
        self.end_headers()


def scan_i2c(i2c: Any) -> list[int]:
    while not i2c.try_lock():
        time.sleep(0.01)

    try:
        return i2c.scan()
    finally:
        i2c.unlock()


def read_sensor_loop() -> None:
    global startup_error

    try:
        import adafruit_mlx90640
        import board
        import busio
    except ImportError as exc:
        startup_error = (
            "Missing MLX90640 dependencies. Install adafruit-circuitpython-mlx90640. "
            f"Original error: {exc}"
        )
        startup_ready.set()
        return

    addresses: list[int] = []

    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)
        addresses = scan_i2c(i2c)

        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = getattr(
            adafruit_mlx90640.RefreshRate,
            SENSOR_REFRESH_RATE,
        )
    except Exception as exc:
        detected = ", ".join(f"0x{address:02x}" for address in addresses) or "none"
        startup_error = (
            "Could not initialize MLX90640 on Raspberry Pi I2C pins "
            f"(SDA GPIO 2, SCL GPIO 3). I2C scan found: {detected}. "
            f"Expected MLX90640 at 0x33. Original error: {exc}"
        )
        startup_ready.set()
        return

    frame = [0.0] * PIXELS

    while True:
        try:
            mlx.getFrame(frame)
            thermal_frame.update([round(value, 2) for value in frame])
            startup_ready.set()
        except ValueError:
            time.sleep(0.02)
            continue
        except Exception as exc:
            print(f"Sensor read error: {exc}", file=sys.stderr)
            time.sleep(0.2)
            continue

        time.sleep(POLL_SECONDS)


def main() -> None:
    sensor_thread = threading.Thread(target=read_sensor_loop, daemon=True)
    sensor_thread.start()

    startup_ready.wait(timeout=3)

    if startup_error is not None:
        print(startup_error, file=sys.stderr)
        raise SystemExit(1)

    print(f"Open: http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
