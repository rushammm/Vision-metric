# Side-by-side raw vs undistorted preview, for sanity-checking calibration.

import argparse
from pathlib import Path

import cv2
import numpy as np


def undistort_demo(image_path, intrinsics_path, out_path):
    data = np.load(intrinsics_path, allow_pickle=True)
    K, dist = data["K"], data["dist"]

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1.0, newImgSize=(w, h))
    und = cv2.undistort(img, K, dist, None, new_K)

    label_h = 60
    canvas = np.zeros((h + label_h, w * 2, 3), dtype=np.uint8)
    canvas[label_h:, :w] = img
    canvas[label_h:, w:] = und
    cv2.putText(canvas, "RAW (distorted)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.putText(canvas, "UNDISTORTED", (w + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=Path("calibration/intrinsics.npz"))
    p.add_argument("--out", type=Path, default=Path("calibration/undistortion_demo.jpg"))
    args = p.parse_args()
    undistort_demo(args.image, args.intrinsics, args.out)
