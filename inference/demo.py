# End-to-end demo: image in, annotated image out (mask + W/H/confidence).

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "measurement"))
from measure import load_model, measure, read_image


def annotate(img, result):
    vis = img.copy()
    H, W = vis.shape[:2]

    # mask overlay
    mask = result["mask"]
    overlay = vis.copy()
    overlay[mask > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # card rect (yellow)
    box = cv2.boxPoints(result["card_rect"]).astype(int)
    cv2.drawContours(vis, [box], 0, (0, 255, 255), max(3, W // 600))

    # notebook rect (green)
    box = cv2.boxPoints(result["notebook_rect"]).astype(int)
    cv2.drawContours(vis, [box], 0, (0, 255, 0), max(4, W // 400))

    # label bar
    bar_h = max(60, H // 20)
    cv2.rectangle(vis, (0, 0), (W, bar_h), (0, 0, 0), -1)
    label = (f"W = {result['width_mm']:.1f} mm   "
             f"H = {result['height_mm']:.1f} mm   "
             f"conf = {result['confidence']:.2f}")
    font_scale = max(0.8, W / 1800)
    cv2.putText(vis, label, (20, int(bar_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), max(2, W // 800))
    return vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--weights", default="models/checkpoints/best.pt")
    p.add_argument("--intrinsics", default="calibration/intrinsics.npz")
    p.add_argument("--out", default="inference/outputs/demo_output.jpg")
    p.add_argument("--auto-card", action="store_true",
                   help="try auto card detection first (default: manual click)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    intr = np.load(args.intrinsics, allow_pickle=True)
    K, dist = intr["K"], intr["dist"]

    model = load_model(args.weights, device)
    img = read_image(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    print(f"running on {args.image} ...")
    r = measure(img, K, dist, model, device, auto_card=args.auto_card)

    print()
    print(f"  width  = {r['width_mm']:.2f} mm")
    print(f"  height = {r['height_mm']:.2f} mm")
    print(f"  confidence = {r['confidence']:.3f}")

    vis = annotate(r["undistorted"], r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
