# Render predicted masks on the test images. Per-image JPEGs + 2x3 contact sheet.

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "measurement"))
from measure import load_model


@torch.no_grad()
def predict(model, pil_img, device, score_thresh):
    t = torchvision.transforms.functional.to_tensor(pil_img).to(device)
    out = model([t])[0]
    if len(out["scores"]) == 0 or float(out["scores"][0]) < score_thresh:
        return None, 0.0
    best = int(out["scores"].argmax())
    score = float(out["scores"][best])
    mask = (out["masks"][best, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return mask, score


def annotate(bgr, mask, gt_mask, score):
    vis = bgr.copy()

    # GT mask outline (white)
    if gt_mask is not None:
        cs, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (255, 255, 255), 4)

    # predicted mask fill (green) + outline
    if mask is not None:
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 3)

    # IoU
    iou = None
    if mask is not None and gt_mask is not None:
        gt_b = gt_mask.astype(bool); pr_b = mask.astype(bool)
        inter = (gt_b & pr_b).sum(); union = (gt_b | pr_b).sum()
        iou = inter / union if union else 0.0

    H, W = vis.shape[:2]
    bar_h = max(60, H // 18)
    cv2.rectangle(vis, (0, 0), (W, bar_h), (0, 0, 0), -1)
    label = f"score={score:.2f}" + (f"   IoU={iou:.3f}" if iou is not None else "")
    cv2.putText(vis, label, (20, int(bar_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W / 1800),
                (255, 255, 255), max(2, W // 800))
    return vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="dataset/splits/test")
    p.add_argument("--ann",       default="dataset/splits/test/_annotations.coco.json")
    p.add_argument("--weights",   default="models/checkpoints/best.pt")
    p.add_argument("--out-dir",   default="docs/figures/test_predictions")
    p.add_argument("--score-thresh", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.ann)
    model = load_model(args.weights, device)

    tiles = []
    for img_id in coco.getImgIds():
        info = coco.imgs[img_id]
        pil = ImageOps.exif_transpose(Image.open(Path(args.data_dir) / info["file_name"])).convert("RGB")
        if pil.size != (info["width"], info["height"]):
            pil = pil.rotate(90, expand=True) if pil.size[::-1] == (info["width"], info["height"]) else pil

        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        mask, score = predict(model, pil, device, args.score_thresh)
        gt_mask = coco.annToMask(coco.loadAnns(coco.getAnnIds(imgIds=img_id))[0])

        vis = annotate(bgr, mask, gt_mask, score)
        out_name = Path(info["file_name"]).stem + "_pred.jpg"
        cv2.imwrite(str(out_dir / out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {info['file_name']}  score={score:.2f}")
        tiles.append(vis)

    # contact sheet (2 rows x 3 cols)
    th, tw = 360, 480
    rows = []
    for r in range(2):
        row = []
        for c in range(3):
            i = r * 3 + c
            t = cv2.resize(tiles[i], (tw, th)) if i < len(tiles) else np.zeros((th, tw, 3), np.uint8)
            row.append(t)
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_dir.parent / "test_predictions_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nsaved {len(tiles)} per-image overlays + contact sheet to {out_dir}")


if __name__ == "__main__":
    main()
