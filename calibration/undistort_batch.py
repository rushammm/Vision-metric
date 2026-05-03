# Undistort every image in a directory using saved intrinsics.

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def undistort_dir(in_dir, out_dir, intrinsics_path):
    data = np.load(intrinsics_path, allow_pickle=True)
    K, dist = data["K"], data["dist"]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in in_dir.iterdir() if p.suffix in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images in {in_dir}")

    new_K_cache = {}
    for p in tqdm(paths, desc="undistort"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip {p.name}")
            continue
        h, w = img.shape[:2]
        if (w, h) not in new_K_cache:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0, newImgSize=(w, h))
            new_K_cache[(w, h)] = new_K
        cv2.imwrite(str(out_dir / p.name),
                    cv2.undistort(img, K, dist, None, new_K_cache[(w, h)]))

    print(f"\nwrote {len(paths)} images to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=Path("calibration/intrinsics.npz"))
    args = p.parse_args()
    undistort_dir(args.in_dir, args.out_dir, args.intrinsics)
