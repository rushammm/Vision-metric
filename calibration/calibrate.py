# Intrinsic camera calibration from checkerboard photos.

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def detect_corners(gray, pattern_size):
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
    if ok:
        return corners, "SB"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, classic_flags)
    if ok:
        sub_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), sub_criteria)
        return corners, "classic"

    return None, None


def calibrate(image_dir, pattern_size, square_size_mm, out_path, debug_dir=None):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_pts = []
    img_pts = []
    used = []

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir}")

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"SKIP  {path.name} (could not read)", flush=True)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            print(f"SKIP  {path.name} (resolution {gray.shape[1]}x{gray.shape[0]} "
                  f"!= {image_size[0]}x{image_size[1]})", flush=True)
            continue

        corners, method = detect_corners(gray, pattern_size)
        if corners is None:
            print(f"SKIP  {path.name} (no corners)", flush=True)
            if debug_dir is not None:
                small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
                cv2.imwrite(str(debug_dir / f"FAIL_{path.stem}.jpg"), small)
            continue

        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(path.name)
        print(f"OK[{method}] {path.name}", flush=True)

        if debug_dir is not None:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            small = cv2.resize(vis, (vis.shape[1] // 4, vis.shape[0] // 4))
            cv2.imwrite(str(debug_dir / f"OK_{path.stem}.jpg"), small)

    if len(obj_pts) < 10:
        raise RuntimeError(
            f"Only {len(obj_pts)} usable images. Need at least 10.")

    def solve(o_pts, i_pts, names, label):
        print(f"\n--- Pass: {label} ({len(o_pts)} images) ---", flush=True)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            o_pts, i_pts, image_size, None, None)
        errs = []
        for i, name in enumerate(names):
            proj, _ = cv2.projectPoints(o_pts[i], rvecs[i], tvecs[i], K, dist)
            err = float(cv2.norm(i_pts[i], proj, cv2.NORM_L2) / len(proj))
            errs.append((name, err))
        mean_err = sum(e for _, e in errs) / len(errs)
        print(f"RMS reprojection error: {rms:.4f} px")
        print(f"Mean per-image error:   {mean_err:.4f} px")
        return rms, K, dist, errs, mean_err

    # Pass 1: fit on every detected image
    rms1, K1, dist1, errs1, mean1 = solve(obj_pts, img_pts, used, "all detected images")

    # Pass 2: drop images whose per-image error > max(1.0 px, 3 * median)
    median_err = float(np.median([e for _, e in errs1]))
    threshold = max(1.0, 3.0 * median_err)
    keep_idx = [i for i, (_, e) in enumerate(errs1) if e <= threshold]
    dropped = [(n, e) for (n, e) in errs1 if e > threshold]

    if dropped and len(keep_idx) >= 10:
        print(f"\nOutlier rejection: threshold = {threshold:.4f} px "
              f"(max of 1.0 and 3 x median {median_err:.4f}).")
        print(f"Dropping {len(dropped)} image(s):")
        for n, e in sorted(dropped, key=lambda t: -t[1]):
            print(f"  drop  err={e:.4f}  {n}")

        obj_pts2 = [obj_pts[i] for i in keep_idx]
        img_pts2 = [img_pts[i] for i in keep_idx]
        used2 = [used[i] for i in keep_idx]
        rms, K, dist, errs, mean_err = solve(obj_pts2, img_pts2, used2,
                                              "after outlier rejection")
        final_used = used2
    else:
        rms, K, dist, errs, mean_err = rms1, K1, dist1, errs1, mean1
        final_used = used
        print("\nNo outliers above threshold; keeping all images.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             K=K,
             dist=dist,
             image_size=np.array(image_size),
             reprojection_error_rms=np.array(rms),
             reprojection_error_mean=np.array(mean_err),
             used_images=np.array(final_used))

    print("\n=== Final calibration results ===")
    print("K =")
    print(K)
    print(f"\ndist (k1, k2, p1, p2, k3) = {dist.ravel()}")
    print(f"\nRMS reprojection error: {rms:.4f} px")
    print(f"Mean per-image error:   {mean_err:.4f} px")
    print(f"Images used:            {len(final_used)}")
    print(f"\nSaved {out_path}")

    print("\nPer-image errors after final pass (worst first):")
    for name, err in sorted(errs, key=lambda t: -t[1]):
        print(f"  {err:.4f}  {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("calibration/images"))
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per col")
    p.add_argument("--square-mm", type=float, required=True)
    p.add_argument("--out", type=Path, default=Path("calibration/intrinsics.npz"))
    p.add_argument("--debug-dir", type=Path, default=Path("calibration/debug"))
    args = p.parse_args()
    calibrate(args.images, (args.cols, args.rows), args.square_mm, args.out,
              debug_dir=args.debug_dir)
