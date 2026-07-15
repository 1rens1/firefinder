#!/usr/bin/env python3
"""HTTP server for the fire/smoke YOLO11n model, accelerated on the IMX500."""

from __future__ import annotations

import argparse
import io
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

DEFAULT_PORT = 8080
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_JPEG_QUALITY = 70
DEFAULT_FRAME_RATE = 16
DEFAULT_MODEL = Path("models/best_imx_model/packerOut.zip")
DEFAULT_LABELS = Path("models/best_imx_model/labels.txt")
HTML_PATH = Path(__file__).with_name("index.html")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class FireSmokeModel(Model):
    def __init__(self, model_file: Path, labels_file: Path) -> None:
        super().__init__(
            model_file=str(model_file),
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = labels_file.read_text().splitlines()

    def post_process(self, output_tensors):  # noqa: ANN001, ANN201
        return pp_od_yolo_ultralytics(output_tensors)


class StreamingOutput(io.BufferedIOBase):
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.cond = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)

    def wake(self) -> None:
        with self.cond:
            self.cond.notify_all()


output = StreamingOutput()
latest_detections: list[Detection] = []
latest_frame_size = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
detections_lock = threading.Lock()
shutdown_event = threading.Event()


def detection_status(detections: list[Detection]) -> str:
    labels = {d.label.lower() for d in detections}

    if "fire" in labels:
        return "Fire detected"
    if "smoke" in labels:
        return "Smoke detected"
    return "Clear"


def read_index_html() -> bytes:
    if HTML_PATH.exists():
        return HTML_PATH.read_bytes()
    return b"<html><body>Missing src/firefinder/index.html</body></html>"


def write_json(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, object],
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/":
            body = read_index_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            try:
                last_frame: bytes | None = None
                while not shutdown_event.is_set():
                    with output.cond:
                        output.cond.wait(timeout=0.5)
                        frame = output.frame

                    if frame is None or frame is last_frame:
                        continue
                    last_frame = frame

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        if path == "/detections":
            with detections_lock:
                detections = list(latest_detections)
                frame_width, frame_height = latest_frame_size
            status_text = detection_status(detections)
            write_json(
                self,
                200,
                {
                    "status": status_text,
                    "frame": {
                        "width": frame_width,
                        "height": frame_height,
                    },
                    "detections": [
                        {
                            "label": d.label,
                            "confidence": round(d.confidence, 3),
                            "box": list(d.bbox_xyxy),
                        }
                        for d in detections
                    ],
                },
            )
            return

        self.send_response(404)
        self.end_headers()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve live fire/smoke detections, accelerated on the IMX500."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--frame-rate", type=int, default=DEFAULT_FRAME_RATE)
    return parser.parse_args()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def scale_detection(
    detection: Detection,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> Detection:
    source_width, source_height = source_size
    target_width, target_height = target_size

    if source_size == target_size:
        return detection

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1, y1, x2, y2 = detection.bbox_xyxy

    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        bbox_xyxy=(
            clamp(round(x1 * scale_x), 0, target_width),
            clamp(round(y1 * scale_y), 0, target_height),
            clamp(round(x2 * scale_x), 0, target_width),
            clamp(round(y2 * scale_y), 0, target_height),
        ),
    )


def capture_loop(
    model: FireSmokeModel,
    threshold: float,
    target_size: tuple[int, int],
    jpeg_quality: int,
    frame_rate: int,
) -> None:
    global latest_detections, latest_frame_size

    device = AiCamera(frame_rate=frame_rate)
    device.deploy(model)

    with device as stream:
        for frame in stream:
            if shutdown_event.is_set():
                break

            image = frame.image
            source_height, source_width = image.shape[:2]
            source_size = (source_width, source_height)

            dets = frame.detections[frame.detections.confidence > threshold]
            detections = [
                Detection(
                    label=str(model.labels[int(class_id)]),
                    confidence=float(score),
                    bbox_xyxy=tuple(int(v) for v in bbox),
                )
                for bbox, score, class_id, _ in dets
            ]
            scaled_detections = [
                scale_detection(detection, source_size, target_size)
                for detection in detections
            ]

            with detections_lock:
                latest_detections = scaled_detections
                latest_frame_size = target_size

            if source_size != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            ok, jpeg = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if ok:
                output.write(jpeg.tobytes())


def main() -> None:
    args = get_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    model = FireSmokeModel(args.model, args.labels)
    target_size = (max(1, args.width), max(1, args.height))
    jpeg_quality = clamp(args.jpeg_quality, 1, 100)
    frame_rate = max(1, args.frame_rate)

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(model, args.threshold, target_size, jpeg_quality, frame_rate),
        daemon=True,
    )
    capture_thread.start()

    server = Server(("0.0.0.0", args.port), Handler)
    print(f"Open: http://0.0.0.0:{args.port}")

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        output.wake()
        server.shutdown()
        server.server_close()
        capture_thread.join(timeout=3)


if __name__ == "__main__":
    main()
