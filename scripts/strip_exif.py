# Strip EXIF (GPS, device serial, timestamps) from every JPEG/HEIC in the repo.
# Re-saves images in place. Run once before pushing to a public repo.

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


JPEG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
PNG_EXTS = {".png", ".PNG"}
HEIC_EXTS = {".heic", ".heif", ".HEIC", ".HEIF"}


def strip_one(path):
    pil = ImageOps.exif_transpose(Image.open(path))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    if path.suffix in HEIC_EXTS:
        pil.save(path, format="HEIF", quality=95)
    elif path.suffix in PNG_EXTS:
        pil.save(path, format="PNG")
    else:
        pil.save(path, format="JPEG", quality=95, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dirs", nargs="+", default=[
        "calibration/images",
        "calibration/debug",
        "dataset/raw",
        "dataset/undistorted",
        "dataset/splits",
        "dataset/splits_small",
        "dataset/notebook-segmentation.coco-segmentation",
        "docs/figures",
        "inference/outputs",
    ])
    args = p.parse_args()

    root = Path(args.root)
    all_exts = JPEG_EXTS | PNG_EXTS | HEIC_EXTS
    n_total = 0
    n_failed = 0
    for d in args.dirs:
        d_path = root / d
        if not d_path.exists():
            continue
        files = [f for f in d_path.rglob("*") if f.suffix in all_exts]
        print(f"{d}: {len(files)} files")
        for f in files:
            try:
                strip_one(f)
                n_total += 1
            except Exception as e:
                print(f"  FAIL {f}: {e}")
                n_failed += 1
    print(f"\nstripped {n_total} files, {n_failed} failed")


if __name__ == "__main__":
    main()
