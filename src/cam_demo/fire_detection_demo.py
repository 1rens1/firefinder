#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from ultralytics import YOLO

DEFAULT_PORT = 8080
DEFAULT_MODEL = Path("models/fire_yolo11n.pt")
CLASS_COLORS = {
    "smoke": (160, 160, 160),
    "fire": (255, 90, 0),
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class StreamingOutput(io.BufferedIOBase):
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.cond = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


output = StreamingOutput()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                <body style="margin:0;background:#111;color:white;font-family:sans-serif">
                <h2 style="margin:16px">FireFinder YOLO11n Camera</h2>
                <img src="/stream" style="display:block;width:100%;max-width:960px">
                </body>
                </html>
                """
            )
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
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

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve live fire/smoke detections from a trained YOLO11n model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train YOLO11n on Colab, then copy best.pt to "
            f"{DEFAULT_MODEL} or pass --model /path/to/best.pt."
        )

    return YOLO(str(model_path))


def detect_frame(
    model: YOLO,
    frame: Any,
    *,
    threshold: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[Detection]:
    model_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model.predict(
        source=model_frame,
        conf=threshold,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    detections: list[Detection] = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = str(result.names.get(class_id, class_id))
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

    return detections


def detection_status(detections: list[Detection]) -> tuple[str, tuple[int, int, int]]:
    labels = {d.label.lower() for d in detections}

    if "fire" in labels:
        if "smoke" in labels:
            return "Fire + smoke detected", (255, 0, 0)
        return "Fire detected", (255, 0, 0)

    if "smoke" in labels:
        return "Smoke detected", (255, 180, 0)

    return "Clear", (0, 180, 0)


def draw_label(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    cv2.rectangle(
        frame,
        (x - 3, y - text_h - baseline - 4),
        (x + text_w + 3, y + baseline),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_detections(frame: Any, detections: list[Detection]) -> Any:
    status_text, status_color = detection_status(detections)
    draw_label(frame, status_text, (12, 28), status_color)

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = CLASS_COLORS.get(det.label.lower(), (0, 255, 0))
        label = f"{det.label} {det.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, label, (x1 + 4, max(y1 - 8, 24)), color)

    return frame


def main() -> None:
    args = get_args()
    model = load_model(args.model)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        buffer_count=4,
    )
    picam2.configure(config)

    def annotate_request(request: Any) -> None:
        with MappedArray(request, "main") as mapped:
            detections = detect_frame(
                model,
                mapped.array,
                threshold=args.threshold,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
            )
            draw_detections(mapped.array, detections)

    picam2.pre_callback = annotate_request

    try:
        picam2.start_recording(MJPEGEncoder(), FileOutput(output))
        print(f"Open: http://0.0.0.0:{args.port}")
        ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    finally:
        picam2.stop_recording()


if __name__ == "__main__":
    main()
