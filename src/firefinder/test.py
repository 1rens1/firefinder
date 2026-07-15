from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_imx_model" / "pack" / "network.rpk"
LABELS_PATH = PROJECT_ROOT / "models" / "best_imx_model" / "labels.txt"

labels = LABELS_PATH.read_text().splitlines()


def get_color(cls):
    colors = {
        0: (0, 255, 255), # smoke: cyan
        1: (255, 50, 0), # fire: orange
    }
    return colors.get(cls)


imx500 = IMX500(str(MODEL_PATH))

intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = intrinsics.task or "object detection"
intrinsics.labels = labels
intrinsics.update_with_defaults()

print("bbox_normalization:", getattr(intrinsics, "bbox_normalization", None))
print("bbox_order:", getattr(intrinsics, "bbox_order", None))

picam2 = Picamera2(imx500.camera_num)

config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    buffer_count=4,
)

picam2.configure(config)
imx500.show_network_fw_progress_bar()
picam2.start()

for i in range(30):
    request = picam2.capture_request()
    metadata = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    if np_outputs is not None and int(np_outputs[3][0][0]) > 0:
        break

    if i < 29:
        request.release()

np_outputs = imx500.get_outputs(metadata, add_batch=True)

if np_outputs is None:
    print("No inference output")
    frame = request.make_array("main")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(CURRENT_DIR / "output.jpg"), frame)
else:
    boxes = np_outputs[0][0]
    scores = np_outputs[1][0]
    classes = np_outputs[2][0]
    num_dets = int(np_outputs[3][0][0])

    print("raw boxes:", boxes[:num_dets])
    print("raw scores:", scores[:num_dets])
    print("raw classes:", classes[:num_dets])

    input_w, input_h = imx500.get_input_size()

    bbox_norm = getattr(intrinsics, "bbox_normalization", False)

    with MappedArray(request, "main") as m:
        for i in range(num_dets):
            score = float(scores[i])
            if score < 0.25:
                continue

            box = boxes[i]
            cls = int(classes[i])

            # Raw output order is [y1, x1, y2, x2], normalize if needed
            if not bbox_norm and box.max() > 1.0:
                box = box / np.array([input_h, input_w, input_h, input_w])

            y1, x1, y2, x2 = box
            scaled = imx500.convert_inference_coords((x1, y1, x2, y2), metadata, picam2)
            x, y, w, h = [int(v) for v in scaled]

            label = labels[cls] if cls < len(labels) else str(cls)
            text = f"{label} {score:.2f}"

            print(f"{text} [{x}, {y}, {w}, {h}]")

            color = get_color(cls)
            outline_color = (0, 0, 0)

            cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            outline_thickness = 6

            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            text_y = max(text_h + 5, y - 5)

            cv2.putText(
                m.array,
                text,
                (x, text_y),
                font,
                font_scale,
                outline_color,
                outline_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                m.array,
                text,
                (x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        frame = cv2.cvtColor(m.array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(CURRENT_DIR / "output.jpg"), frame)

request.release()
picam2.stop()
