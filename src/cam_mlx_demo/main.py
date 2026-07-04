#!/usr/bin/env python3

import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 8080
HTML_PATH = Path(__file__).with_name("index.html")
WIDTH = 32
HEIGHT = 24
PIXELS = WIDTH * HEIGHT
I2C_FREQUENCY = 800000
SENSOR_REFRESH_RATE = "REFRESH_4_HZ"
POLL_SECONDS = 0.25
FRAME_RETRY_SECONDS = 0.05
SENSOR_ERROR_LOG_SECONDS = 5.0
SENSOR_RESET_ERRORS = 5
SENSOR_RESET_SECONDS = 1.0


class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame: bytes | None = None
        self.cond = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


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

    def snapshot(self) -> dict[str, object] | None:
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


output = Output()
thermal_frame = ThermalFrame()
startup_ready = threading.Event()
startup_error: str | None = None

INDEX_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Firefinder</title></head>
<body><p>Missing src/cam_mlx_demo/index.html</p></body>
</html>
"""


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


def read_index_html() -> bytes:
    if HTML_PATH.exists():
        return HTML_PATH.read_bytes()
    return INDEX_HTML.encode("utf-8")


def reload_version() -> int:
    paths = [Path(__file__), HTML_PATH]
    return max((path.stat().st_mtime_ns for path in paths if path.exists()), default=0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            body = read_index_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/reload-version":
            write_json(self, 200, {"version": reload_version()})
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                while True:
                    with output.cond:
                        output.cond.wait()
                        frame = output.frame

                    if frame is None:
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

            return

        if self.path == "/thermal":
            frame = thermal_frame.snapshot()
            if frame is None:
                write_json(self, 503, {"error": "No thermal frame is ready yet"})
                return
            write_json(self, 200, frame)
            return

        self.send_response(404)
        self.end_headers()


def scan_i2c(i2c: object) -> list[int]:
    while not i2c.try_lock():
        time.sleep(0.01)
    try:
        return i2c.scan()
    finally:
        i2c.unlock()


def initialize_mlx_sensor(
    adafruit_mlx90640: object,
    board: object,
    busio: object,
) -> tuple[object, object]:
    i2c = None
    addresses: list[int] = []

    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY)
        addresses = scan_i2c(i2c)

        mlx = adafruit_mlx90640.MLX90640(i2c)
        mlx.refresh_rate = getattr(adafruit_mlx90640.RefreshRate, SENSOR_REFRESH_RATE)
        return i2c, mlx
    except Exception as exc:
        if i2c is not None and hasattr(i2c, "deinit"):
            i2c.deinit()

        detected = ", ".join(f"0x{address:02x}" for address in addresses) or "none"
        raise RuntimeError(
            "Could not initialize MLX90640 on Raspberry Pi I2C pins "
            f"(SDA GPIO 2, SCL GPIO 3). I2C scan found: {detected}. "
            f"Expected MLX90640 at 0x33. Original error: {exc}"
        ) from exc


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

    try:
        i2c, mlx = initialize_mlx_sensor(adafruit_mlx90640, board, busio)
    except Exception as exc:
        startup_error = str(exc)
        startup_ready.set()
        return

    frame = [0.0] * PIXELS
    consecutive_errors = 0
    last_error_log = 0.0

    while True:
        try:
            mlx.getFrame(frame)
            thermal_frame.update([round(value, 2) for value in frame])
            consecutive_errors = 0
            startup_ready.set()
        except ValueError:
            time.sleep(FRAME_RETRY_SECONDS)
            continue
        except Exception as exc:
            consecutive_errors += 1
            now = time.time()

            if (
                consecutive_errors == 1
                or now - last_error_log >= SENSOR_ERROR_LOG_SECONDS
            ):
                print(
                    f"Sensor read error ({consecutive_errors} consecutive): {exc}",
                    file=sys.stderr,
                )
                last_error_log = now

            if consecutive_errors >= SENSOR_RESET_ERRORS:
                print("Resetting MLX90640 after repeated read errors", file=sys.stderr)

                if hasattr(i2c, "deinit"):
                    try:
                        i2c.deinit()
                    except Exception:
                        pass

                time.sleep(SENSOR_RESET_SECONDS)

                try:
                    i2c, mlx = initialize_mlx_sensor(
                        adafruit_mlx90640,
                        board,
                        busio,
                    )
                    consecutive_errors = 0
                    print("MLX90640 reset complete", file=sys.stderr)
                except Exception as reset_exc:
                    print(f"MLX90640 reset failed: {reset_exc}", file=sys.stderr)
                    time.sleep(SENSOR_RESET_SECONDS)
            else:
                time.sleep(min(1.0, FRAME_RETRY_SECONDS * consecutive_errors))

            continue

        time.sleep(POLL_SECONDS)


def main() -> None:
    sensor_thread = threading.Thread(target=read_sensor_loop, daemon=True)
    sensor_thread.start()

    startup_ready.wait(timeout=3)

    if startup_error is not None:
        print(startup_error, file=sys.stderr)
        raise SystemExit(1)

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    print(f"Open: http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
