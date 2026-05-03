# Fine-tune Mask R-CNN (ResNet-50 FPN, COCO-pretrained) on the notebook dataset.
# Saves best.pt whenever val mask IoU improves.

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from pycocotools.coco import COCO


class NotebookDataset(data.Dataset):
    def __init__(self, root, ann_file, train=False):
        self.root = Path(root)
        self.coco = COCO(ann_file)
        self.ids = sorted(i for i in self.coco.imgs
                          if self.coco.getAnnIds(imgIds=i))
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        c = self.coco
        img_id = self.ids[idx]
        info = c.imgs[img_id]
        img = Image.open(self.root / info["file_name"]).convert("RGB")

        anns = c.loadAnns(c.getAnnIds(imgIds=img_id))
        boxes, masks = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            masks.append(c.annToMask(a))

        # random horizontal flip
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            W = info["width"]
            boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]
            masks = [np.fliplr(m).copy() for m in masks]

        img_t = torchvision.transforms.functional.to_tensor(img)
        target = {
            "boxes":    torch.tensor(boxes, dtype=torch.float32),
            "labels":   torch.ones(len(anns), dtype=torch.int64),
            "masks":    torch.tensor(np.stack(masks), dtype=torch.uint8),
            "image_id": torch.tensor([img_id]),
            "area":     torch.tensor([a["area"] for a in anns], dtype=torch.float32),
            "iscrowd":  torch.zeros(len(anns), dtype=torch.int64),
        }
        return img_t, target


def get_model(num_classes=2):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, 256, num_classes)
    return model


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def eval_iou(model, loader, device):
    model.eval()
    ious = []
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        outs = model(imgs)
        for out, tgt in zip(outs, targets):
            gt = tgt["masks"][0].numpy().astype(bool)
            if len(out["masks"]) == 0:
                ious.append(0.0)
                continue
            best = out["scores"].argmax()
            pr = (out["masks"][best, 0] > 0.5).cpu().numpy()
            inter = (pr & gt).sum()
            union = (pr | gt).sum()
            ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="dataset/splits")
    p.add_argument("--out-dir",    default="models/checkpoints")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = Path(args.data_dir)

    train_ds = NotebookDataset(d / "train", d / "train" / "_annotations.coco.json", train=True)
    val_ds   = NotebookDataset(d / "val",   d / "val"   / "_annotations.coco.json", train=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=collate, num_workers=args.workers)
    val_loader   = data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                    collate_fn=collate, num_workers=args.workers)

    model = get_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.5)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_iou,seconds\n")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        t0 = time.time()
        for imgs, targets in train_loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, targets)
            loss = sum(losses.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss); n += 1
        avg = total / max(n, 1)
        dt = time.time() - t0

        iou = eval_iou(model, val_loader, device)
        sched.step()

        print(f"epoch {ep:>2}: loss={avg:.4f}  val_iou={iou:.4f}  ({dt:.1f}s)")
        with open(log_path, "a") as f:
            f.write(f"{ep},{avg:.6f},{iou:.6f},{dt:.1f}\n")

        torch.save(model.state_dict(), out_dir / "last.pt")
        if iou > best:
            best = iou
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> new best, saved best.pt")

    print(f"\ndone. best val IoU: {best:.4f}")


if __name__ == "__main__":
    main()
