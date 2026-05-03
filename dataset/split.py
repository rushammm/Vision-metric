# Split the Roboflow COCO export into train/val/test (70/20/10).
# Roboflow gave us a single train/ folder so we redo the split ourselves
# with a fixed seed for reproducibility.

import argparse
import json
import random
import shutil
from pathlib import Path


def split_coco(in_dir, out_dir, seed=42, fractions=(0.70, 0.20, 0.10)):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    with open(in_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    images = coco["images"]
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, imgs in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # remap image and annotation ids so each split is self-contained
        new_images = []
        new_anns = []
        next_ann_id = 1
        for new_img_id, im in enumerate(imgs, start=1):
            old_id = im["id"]
            new_im = dict(im)
            new_im["id"] = new_img_id
            new_images.append(new_im)
            for a in anns_by_img.get(old_id, []):
                na = dict(a)
                na["id"] = next_ann_id
                na["image_id"] = new_img_id
                # Roboflow leaves bbox values as strings sometimes; cast to float
                na["bbox"] = [float(x) for x in na["bbox"]]
                na["area"] = float(na["area"])
                new_anns.append(na)
                next_ann_id += 1

            shutil.copy2(in_dir / im["file_name"], split_dir / im["file_name"])

        out_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": new_images,
            "annotations": new_anns,
        }
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(out_coco, f)

        print(f"{split}: {len(new_images)} images, {len(new_anns)} annotations")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="dataset/notebook-segmentation.coco-segmentation/train")
    p.add_argument("--out-dir", default="dataset/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    split_coco(args.in_dir, args.out_dir, seed=args.seed)
