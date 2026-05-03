# Downsize the splits/ images so the upload to Colab is fast.
# Mask R-CNN resizes to ~800px internally anyway, so 1280px max is fine.
# Rewrites the COCO JSON with rescaled bbox / segmentation coordinates.

import argparse
import copy
import json
from pathlib import Path

import cv2


def resize_split(in_dir, out_dir, max_side):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    scales = {}
    new_images = []
    for im in coco["images"]:
        path = in_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            print("skip unreadable", path)
            continue
        h, w = img.shape[:2]
        s = max_side / max(w, h)
        if s >= 1.0:
            cv2.imwrite(str(out_dir / im["file_name"]), img)
            new_w, new_h, s = w, h, 1.0
        else:
            new_w = int(round(w * s))
            new_h = int(round(h * s))
            small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / im["file_name"]), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        scales[im["id"]] = s
        new_im = dict(im)
        new_im["width"] = new_w
        new_im["height"] = new_h
        new_images.append(new_im)

    new_anns = []
    for a in coco["annotations"]:
        s = scales[a["image_id"]]
        na = copy.deepcopy(a)
        na["bbox"] = [c * s for c in a["bbox"]]
        na["area"] = a["area"] * s * s
        na["segmentation"] = [[c * s for c in poly] for poly in a["segmentation"]]
        new_anns.append(na)

    out_coco = dict(coco)
    out_coco["images"] = new_images
    out_coco["annotations"] = new_anns
    with open(out_dir / "_annotations.coco.json", "w") as f:
        json.dump(out_coco, f)
    print(f"{in_dir.name}: {len(new_images)} images written")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="dataset/splits")
    p.add_argument("--out-dir", default="dataset/splits_small")
    p.add_argument("--max-side", type=int, default=1280)
    args = p.parse_args()
    for split in ("train", "val", "test"):
        resize_split(Path(args.in_dir) / split,
                     Path(args.out_dir) / split,
                     args.max_side)
