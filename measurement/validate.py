# Run the measurement pipeline on a list of images and compare to ground truth.
# Reports MAE and MPE for width and height.

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import load_model, measure, read_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", required=True,
                   help="CSV with columns: filename,width_mm,height_mm")
    p.add_argument("--images-dir", default="dataset/raw")
    p.add_argument("--weights", default="models/checkpoints/best.pt")
    p.add_argument("--intrinsics", default="calibration/intrinsics.npz")
    p.add_argument("--out", default="measurement/outputs/results.csv")
    p.add_argument("--auto-card", action="store_true",
                   help="try auto card detection first (default: manual click per image)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    intr = np.load(args.intrinsics, allow_pickle=True)
    K, dist = intr["K"], intr["dist"]
    model = load_model(args.weights, device)

    rows = []
    with open(args.ground_truth) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            true_w = float(row["width_mm"])
            true_h = float(row["height_mm"])

            img_path = Path(args.images_dir) / fn
            try:
                img = read_image(img_path)
            except Exception as e:
                print(f"skip {fn}: {e}")
                continue

            print(f"\n>> {fn}")
            try:
                r = measure(img, K, dist, model, device, auto_card=args.auto_card)
            except Exception as e:
                print(f"  fail: {e}")
                continue

            pw, ph = r["width_mm"], r["height_mm"]
            ew, eh = abs(pw - true_w), abs(ph - true_h)
            pew = ew / true_w * 100
            peh = eh / true_h * 100
            print(f"  pred = ({pw:.1f}, {ph:.1f})  true = ({true_w:.1f}, {true_h:.1f})  "
                  f"err = ({ew:.1f}, {eh:.1f}) mm  ({pew:.2f}%, {peh:.2f}%)")

            rows.append({
                "filename":     fn,
                "true_w_mm":    true_w,
                "pred_w_mm":    round(pw, 2),
                "abs_err_w_mm": round(ew, 2),
                "pct_err_w":    round(pew, 2),
                "true_h_mm":    true_h,
                "pred_h_mm":    round(ph, 2),
                "abs_err_h_mm": round(eh, 2),
                "pct_err_h":    round(peh, 2),
                "confidence":   round(r["confidence"], 3),
            })

    if not rows:
        print("no successful measurements")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mae_w = float(np.mean([r["abs_err_w_mm"] for r in rows]))
    mae_h = float(np.mean([r["abs_err_h_mm"] for r in rows]))
    mpe_w = float(np.mean([r["pct_err_w"] for r in rows]))
    mpe_h = float(np.mean([r["pct_err_h"] for r in rows]))

    print()
    print(f"N = {len(rows)}")
    print(f"Width  MAE = {mae_w:.2f} mm   MPE = {mpe_w:.2f} %")
    print(f"Height MAE = {mae_h:.2f} mm   MPE = {mpe_h:.2f} %")
    print(f"\nresults saved to {args.out}")


if __name__ == "__main__":
    main()
