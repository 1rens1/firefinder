from pathlib import Path
import time

from picamera2 import Picamera2
from picamera2.devices import IMX500
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_imx_model" / "pack" / "network.rpk"

OUTPUT_IMAGE = CURRENT_DIR / "test_capture.jpg"


def main():
    print(f"Loading model: {MODEL_PATH}")

    imx500 = IMX500(str(MODEL_PATH))

    picam2 = Picamera2(imx500.camera_num)

    config = picam2.create_preview_configuration(main={"size": (640, 480)})

    picam2.configure(config)

    picam2.start()

    print("Camera started")

    # time.sleep(2)

    request = picam2.capture_request()

    metadata = request.get_metadata()

    # print("\nMetadata keys:")
    # print(metadata.keys())

    # print("\nCNN output tensor length:")
    # print(len(metadata["CnnOutputTensor"]))

    # print("\nFirst 50 tensor values:")
    # print(metadata["CnnOutputTensor"][:50])

    frame = request.make_array("main")

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)

    boxes = (
        metadata.get("InferenceBoxes")
        or metadata.get("BoundingBoxes")
        or metadata.get("Boxes")
    )
    if boxes:
        for box in boxes:
            if len(box) >= 4:
                draw.rectangle(box[:4], outline=(255, 0, 0), width=2)
    else:
        draw.text((10, 10), "No detection", font=font, fill=(255, 255, 255), stroke_fill=(0, 128, 0), stroke_width=4)

    image.convert("RGB").save(OUTPUT_IMAGE)

    print(f"\nSaved image: {OUTPUT_IMAGE}")

    request.release()
    picam2.stop()


if __name__ == "__main__":
    main()
