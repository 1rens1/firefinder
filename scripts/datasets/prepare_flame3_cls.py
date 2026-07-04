from pathlib import Path
import random
import shutil

SRC = Path("datasets/flame3/FLAME 3 CV Dataset (Sycan Marsh)")
OUT = Path("datasets/flame3_cls")

CLASSES = {
    "Fire": SRC / "Fire" / "RGB" / "Corrected FOV",
    "No Fire": SRC / "No Fire" / "RGB" / "Corrected FOV",
}

TRAIN_RATIO = 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

for class_name, src_dir in CLASSES.items():
    images = [p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    splits = {
        "train": images[:split_idx],
        "val": images[split_idx:],
    }

    safe_class = class_name.lower().replace(" ", "_")

    for split, paths in splits.items():
        out_dir = OUT / split / safe_class
        out_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

        print(split, safe_class, len(paths))

print(f"Done: {OUT}")
