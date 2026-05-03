# Evaluate trained Mask R-CNN on the test split.
# Reports COCO mAP (segm + bbox) and precision/recall/F1 at IoU 0.5.

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "measurement"))
from measure import load_model


def encode_mask(m):
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@torch.no_grad()
def run_inference(model, coco, img_dir, device, score_thresh):
    preds = []
    for img_id in coco.getImgIds():
        info = coco.imgs[img_id]
        img = ImageOps.exif_transpose(Image.open(img_dir / info["file_name"])).convert("RGB")
        # Roboflow labelled in EXIF-rotated orientation; rotate to match
        # so the polygon coords line up.
        if img.size != (info["width"], info["height"]):
            if img.size[::-1] == (info["width"], info["height"]):
                img = img.rotate(90, expand=True)
        t = torchvision.transforms.functional.to_tensor(img).to(device)
        out = model([t])[0]

        for i in range(len(out["scores"])):
            score = float(out["scores"][i])
            if score < score_thresh:
                continue
            mask = (out["masks"][i, 0] > 0.5).cpu().numpy()
            x1, y1, x2, y2 = out["boxes"][i].cpu().numpy().tolist()
            preds.append({
                "image_id":     img_id,
                "category_id":  int(out["labels"][i]),
                "bbox":         [x1, y1, x2 - x1, y2 - y1],
                "score":        score,
                "segmentation": encode_mask(mask),
            })
    return preds


def coco_eval(coco, preds, iou_type):
    if not preds:
        return {"mAP": 0.0, "mAP50": 0.0}
    coco_dt = coco.loadRes(preds)
    e = COCOeval(coco, coco_dt, iou_type)
    with redirect_stdout(io.StringIO()):
        e.evaluate(); e.accumulate(); e.summarize()
    return {"mAP": float(e.stats[0]), "mAP50": float(e.stats[1])}


def precision_recall_f1(coco, preds, iou_thresh, score_thresh):
    # one ground-truth instance per image; pick the highest-scoring
    # prediction above score_thresh; TP if mask IoU >= iou_thresh, else FP.
    # No prediction above threshold while a GT exists -> FN.
    tp = fp = fn = 0
    for img_id in coco.getImgIds():
        gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gt_mask = coco.annToMask(gt_anns[0]).astype(bool)

        cands = [p for p in preds if p["image_id"] == img_id and p["score"] >= score_thresh]
        if not cands:
            fn += 1
            continue
        best = max(cands, key=lambda p: p["score"])
        pr_mask = mask_utils.decode(best["segmentation"]).astype(bool)
        inter = (pr_mask & gt_mask).sum()
        union = (pr_mask | gt_mask).sum()
        iou = inter / union if union else 0.0
        if iou >= iou_thresh:
            tp += 1
        else:
            fp += 1
            fn += 1  # GT was not matched by a prediction with sufficient overlap

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="dataset/splits/test")
    p.add_argument("--ann",       default="dataset/splits/test/_annotations.coco.json")
    p.add_argument("--weights",   default="models/checkpoints/best.pt")
    p.add_argument("--out",       default="models/checkpoints/test_metrics.json")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--iou-thresh",   type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    coco = COCO(args.ann)
    model = load_model(args.weights, device)
    print(f"running on {len(coco.getImgIds())} test images...")
    preds = run_inference(model, coco, Path(args.data_dir), device, args.score_thresh)
    print(f"got {len(preds)} predictions above score {args.score_thresh}")

    bbox = coco_eval(coco, preds, "bbox")
    segm = coco_eval(coco, preds, "segm")
    prec, rec, f1, tp, fp, fn = precision_recall_f1(
        coco, preds, args.iou_thresh, args.score_thresh)

    metrics = {
        "n_test_images":   len(coco.getImgIds()),
        "score_threshold": args.score_thresh,
        "iou_threshold":   args.iou_thresh,
        "bbox_mAP":        round(bbox["mAP"], 4),
        "bbox_mAP50":      round(bbox["mAP50"], 4),
        "segm_mAP":        round(segm["mAP"], 4),
        "segm_mAP50":      round(segm["mAP50"], 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "f1":              round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }

    print()
    for k, v in metrics.items():
        print(f"  {k:>16s}: {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
