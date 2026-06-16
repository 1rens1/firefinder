#!/usr/bin/env python3
"""Download and extract the FLAME 3 Computer Vision subset from Kaggle.

Usage:
    uv run python scripts/download_flame3.py
    uv run python scripts/download_flame3.py --force
    uv run python scripts/download_flame3.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "brycehopkins/flame-3-computer-vision-subset-sycan-marsh"
ZIP_NAME = "flame-3-computer-vision-subset-sycan-marsh.zip"
DEFAULT_OUT_DIR = Path("datasets/flame3")


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        print(
            "Kaggle CLI not found. Install it with:\n"
            "  uv add kaggle\n\n"
            "Then add your Kaggle token to ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        raise SystemExit(1)


def download_dataset(zip_path: Path, force: bool) -> None:
    if zip_path.exists() and not force:
        print(f"Found existing {zip_path}. Use --force to re-download.")
        return

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(zip_path.parent),
    ]
    if force:
        cmd.append("--force")

    run(cmd)


def extract_dataset(zip_path: Path, out_dir: Path, clean: bool) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print("Done.")
    print(f"Dataset extracted to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--zip", type=Path, default=Path(ZIP_NAME))
    parser.add_argument("--force", action="store_true", help="Re-download the Kaggle zip")
    parser.add_argument("--clean", action="store_true", help="Delete output folder before extracting")
    args = parser.parse_args()

    ensure_kaggle_cli()
    download_dataset(args.zip, args.force)
    extract_dataset(args.zip, args.out, args.clean)


if __name__ == "__main__":
    main()
