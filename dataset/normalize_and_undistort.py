# Pre-process raw photos for labelling: handle HEIC + JPEG, EXIF rotate to
# landscape, scale K to match each file's resolution (assumes uniform
# downscale), and undistort. Output goes to dataset/undistorted/.

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()


READABLE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def scale_K(K, scale):
    K2 = K.copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] *= scale
    K2[1, 2] *= scale
    return K2


def load_landscape_bgr(path):
    pil = ImageOps.exif_transpose(Image.open(path))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    w, h = pil.size
    rotated = False
    if h > w:
        pil = pil.rotate(-90, expand=True)
        rotated = True

    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return bgr, ("portrait_rot" if rotated else "landscape")


def normalize_and_undistort(in_dir, out_dir, intrinsics_path, calib_size):
    data = np.load(intrinsics_path, allow_pickle=True)
    K_calib = data["K"]
    dist = data["dist"]
    calib_w, calib_h = calib_size
    expected_aspect = calib_w / calib_h

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in READABLE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images in {in_dir}")

    K_cache = {}
    bad_aspect = []
    written = 0

    for p in tqdm(paths, desc="normalise+undistort"):
        try:
            bgr, kind = load_landscape_bgr(p)
        except Exception as e:
            print(f"  unreadable {p.name}: {e}")
            continue

        h, w = bgr.shape[:2]
        aspect = w / h
        if abs(aspect - expected_aspect) > 0.01:
            bad_aspect.append((p.name, w, h))
            continue

        if (w, h) not in K_cache:
            scale = w / calib_w
            K_scaled = scale_K(K_calib, scale)
            new_K, _ = cv2.getOptimalNewCameraMatrix(K_scaled, dist, (w, h),
                                                     alpha=1.0, newImgSize=(w, h))
            K_cache[(w, h)] = (K_scaled, new_K)
        K_scaled, new_K = K_cache[(w, h)]

        undistorted = cv2.undistort(bgr, K_scaled, dist, None, new_K)

        out_name = p.stem + ".jpg"
        cv2.imwrite(str(out_dir / out_name), undistorted,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1

    print(f"\nwrote {written} images to {out_dir}")
    print(f"K scaled for {len(K_cache)} resolutions:")
    for (w, h), (K_s, _) in K_cache.items():
        print(f"  {w}x{h}: scale={w/calib_w:.4f}, fx={K_s[0,0]:.1f}")
    if bad_aspect:
        print(f"\nskipped {len(bad_aspect)} images with bad aspect ratio (probably cropped):")
        for name, w, h in bad_aspect:
            print(f"  {name}: {w}x{h}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path("dataset/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("dataset/undistorted"))
    p.add_argument("--intrinsics", type=Path, default=Path("calibration/intrinsics.npz"))
    p.add_argument("--calib-w", type=int, default=4032)
    p.add_argument("--calib-h", type=int, default=3024)
    args = p.parse_args()
    normalize_and_undistort(args.in_dir, args.out_dir, args.intrinsics,
                            (args.calib_w, args.calib_h))
