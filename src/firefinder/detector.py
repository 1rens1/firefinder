from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: list[float]


class FireDetector:
    def __init__(
        self,
        model_path: str | Path = "models/fire_yolov8n.pt",
        confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Train/export a model first, or use yolov8n.pt only for testing."
            )

        self.model = YOLO(str(self.model_path))

    def detect_frame(self, frame: Any) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0].tolist()]

                detections.append(
                    Detection(
                        class_id=class_id,
                        label=names[class_id],
                        confidence=confidence,
                        bbox_xyxy=bbox,
                    )
                )

        return detections

    def detect_image(self, image_path: str | Path) -> list[Detection]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.detect_frame(image)


def draw_detections(frame: Any, detections: list[Detection]) -> Any:
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        label = f"{det.label} {det.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


if __name__ == "__main__":
    detector = FireDetector(
        model_path="models/fire_yolov8n.pt",
        confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect_frame(frame)
        frame = draw_detections(frame, detections)

        cv2.imshow("FireFinder Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
