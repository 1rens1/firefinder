#!/usr/bin/env python3

import argparse
import io
import sys
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 8080

last_results = []
last_detections = []


class Output(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.cond = threading.Condition()

    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


output = Output()


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata):
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()

    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=args.threshold,
            iou_thres=args.iou,
            max_out_dets=args.max_detections,
        )[0]

        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)

    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        if intrinsics.bbox_normalization:
            boxes = boxes / input_h

        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > args.threshold
    ]

    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    detections = last_results
    labels = get_labels()

    with MappedArray(request, stream) as m:
        cv2.putText(
            m.array,
            "Fire-Finder IMX500",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for detection in detections:
            x, y, w, h = detection.box
            x, y, w, h = int(x), int(y), int(w), int(h)

            label_id = int(detection.category)
            label_name = labels[label_id] if label_id < len(labels) else str(label_id)
            label = f"{label_name} ({detection.conf:.2f})"

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_x = x + 5
            text_y = max(y + 18, 18)

            cv2.rectangle(
                m.array,
                (text_x - 2, text_y - 15),
                (text_x + 180, text_y + 5),
                (255, 255, 255),
                cv2.FILLED,
            )

            cv2.putText(
                m.array,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="margin:0;background:#111;color:white;font-family:sans-serif">
                <h2>Fire-Finder AI Camera</h2>
                <img src="/stream" style="width:100%;max-width:960px">
                </body>
                </html>
                """)
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--labels", default="assets/coco_labels.txt")
    return parser.parse_args()


def main() -> None:
    global args, imx500, intrinsics, picam2

    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics

    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    if intrinsics.labels is None:
        with open(args.labels, "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={"size": (640, 480)},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )

    imx500.show_network_fw_progress_bar()

    picam2.pre_callback = draw_detections
    picam2.configure(config)
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    print(f"Open: http://0.0.0.0:{PORT}")

    def inference_loop():
        global last_results
        while True:
            last_results = parse_detections(picam2.capture_metadata())

    threading.Thread(target=inference_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
