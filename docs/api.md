# API / Module Documentation

Public functions in each module, with signatures, inputs, outputs, and example usage.

---

## `measurement/measure.py`

The core measurement pipeline. Imported by `measurement/validate.py`, `inference/demo.py`, `models/eval_test.py`, `models/viz_test.py`.

### `read_image(path) -> np.ndarray`

Read a JPEG / PNG / HEIC / HEIF file. Apply EXIF orientation. Rotate portrait to landscape so the calibrated K applies.

| Input | Type | Notes |
|---|---|---|
| `path` | `str` or `pathlib.Path` | image path |

| Returns | Shape | dtype |
|---|---|---|
| BGR image | (H, W, 3) | uint8 |

```python
img = read_image("dataset/raw/IMG_4326.JPG.jpeg")
# img.shape == (3024, 4032, 3)
```

### `load_model(weights_path, device) -> torch.nn.Module`

Build a Mask R-CNN ResNet-50 FPN with 2-class heads, load fine-tuned weights, move to device, switch to eval mode.

| Input | Type | Notes |
|---|---|---|
| `weights_path` | `str` or `Path` | path to `best.pt` |
| `device` | `torch.device` | `cuda` or `cpu` |

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("models/checkpoints/best.pt", device)
```

### `undistort(img, K, dist) -> np.ndarray`

Scale K to the input resolution and undistort.

| Input | Type | Notes |
|---|---|---|
| `img` | (H, W, 3) uint8 BGR | input image |
| `K` | (3, 3) float | intrinsic matrix at calibration resolution |
| `dist` | (5,) float | distortion coefficients (k1, k2, p1, p2, k3) |

Returns the undistorted image, same shape as input.

### `click_card_corners(img) -> np.ndarray | None`

Open a window showing a downscaled preview. User clicks the 4 card corners (any order), presses ENTER. Returns 4 points in **original image coordinates**, ordered TL, TR, BR, BL. Returns `None` if user pressed ESC.

| Returns | Shape | dtype |
|---|---|---|
| Ordered corners | (4, 2) | float32 |

```python
corners = click_card_corners(undistorted_img)
# corners[0] = top-left, corners[1] = top-right,
# corners[2] = bottom-right, corners[3] = bottom-left
```

### `card_homography(corners) -> np.ndarray`

Build a homography from card pixels to a metric plane where the card's long side is exactly 85.60 mm. Long-side orientation is auto-detected from the clicked quad's edge lengths.

| Input | Shape | Notes |
|---|---|---|
| `corners` | (4, 2) | TL/TR/BR/BL ordered (output of `click_card_corners` / `order_corners_tl_tr_br_bl`) |

| Returns | Shape | dtype |
|---|---|---|
| Homography | (3, 3) | float64 |

### `predict_notebook_mask(model, img_bgr, device, score_thresh=0.5) -> tuple[np.ndarray | None, float]`

Run Mask R-CNN on one image. Return the highest-score mask (binary) and its score.

Returns `(None, 0.0)` if no detections; `(None, score)` if best score is below threshold.

### `measure(img_bgr, K, dist, model, device, auto_card=False) -> dict`

End-to-end: undistort → click card → homography → mask → contour → mm → minAreaRect.

| Returns key | Type | Description |
|---|---|---|
| `width_mm` | float | long side of the rotated rectangle |
| `height_mm` | float | short side of the rotated rectangle |
| `confidence` | float | mask score (0-1) |
| `card_corners` | (4, 2) float32 | clicked corners in pixels |
| `card_rect` | OpenCV rect | minAreaRect of card (for visualisation) |
| `notebook_rect` | OpenCV rect | minAreaRect of mask in pixels (for visualisation) |
| `mask` | (H, W) uint8 | predicted notebook mask |
| `undistorted` | (H, W, 3) uint8 BGR | undistorted input |
| `homography` | (3, 3) float64 | the H matrix |

```python
intr = np.load("calibration/intrinsics.npz")
model = load_model("models/checkpoints/best.pt", device)
img = read_image("photo.jpg")
result = measure(img, intr["K"], intr["dist"], model, device)
print(f"{result['width_mm']:.1f} x {result['height_mm']:.1f} mm")
# 195.3 x 134.2 mm
```

---

## `inference/demo.py`

CLI wrapper around `measure()` that produces an annotated JPEG.

```
python inference/demo.py --image <path> [--out <path>] [--weights <path>] [--intrinsics <path>]
```

Flags:
- `--image` (required): path to input photo
- `--out` (default `inference/outputs/demo_output.jpg`): annotated output path
- `--weights` (default `models/checkpoints/best.pt`)
- `--intrinsics` (default `calibration/intrinsics.npz`)
- `--auto-card`: try auto card detection first (default: manual click)

Output: an annotated JPEG with the predicted mask filled in green, the card box outlined in yellow, and a black label bar at the top showing `W = ... mm   H = ... mm   conf = ...`.

---

## `measurement/validate.py`

Run the measurement pipeline on a list of images and compare to ground truth.

```
python measurement/validate.py --ground-truth <csv> [--images-dir <dir>] [--out <csv>]
```

Ground-truth CSV format:
```
filename,width_mm,height_mm
IMG_4326.JPG.jpeg,180,125
IMG_4327.JPG.jpeg,180,125
```

Outputs `measurement/outputs/results.csv` with per-image errors and prints aggregate MAE/MPE for width and height.

---

## `models/train.py`

Fine-tune Mask R-CNN on the notebook dataset.

```
python models/train.py [--data-dir dataset/splits] [--epochs 10] [--batch-size 2] [--lr 5e-3]
```

Reads COCO-format annotations from `<data-dir>/{train,val}/_annotations.coco.json`. Logs per-epoch loss and val IoU to `models/checkpoints/training_log.csv`. Saves `last.pt` every epoch and `best.pt` whenever val IoU improves.

---

## `models/eval_test.py`

Evaluate trained weights on the held-out test split.

```
python models/eval_test.py [--data-dir dataset/splits/test] [--ann <coco.json>] [--weights <pt>]
```

Reports segm/bbox mAP@0.5:0.95, mAP@0.5 (via pycocotools) and precision/recall/F1 at IoU 0.5. Writes `models/checkpoints/test_metrics.json`.

---

## `models/viz_test.py`

Render predicted-mask overlays on the test images. Outputs per-image annotated JPEGs and a 2×3 contact sheet.

```
python models/viz_test.py [--data-dir dataset/splits/test] [--out-dir docs/figures/test_predictions]
```

---

## `calibration/calibrate.py`

Run intrinsic calibration on a folder of checkerboard photos.

```
python calibration/calibrate.py --square-mm <float> --cols <int> --rows <int>
```

Args (most-used):
- `--square-mm`: physical size of one square (e.g. 24.0)
- `--cols`, `--rows`: inner-corner counts (e.g. 9 and 6 for a 10x7 board)
- `--in-dir` (default `calibration/images`): folder with the photos

Writes `calibration/intrinsics.npz` with keys `K`, `dist`, `image_size`.

---

## `dataset/normalize_and_undistort.py`

Pre-process raw photos for labelling: read any of {JPEG, PNG, HEIC, HEIF}, apply EXIF orientation, rotate portrait to landscape, scale K to the file's resolution, undistort, save as JPEG.

```
python dataset/normalize_and_undistort.py [--in-dir dataset/raw] [--out-dir dataset/undistorted]
```

---

## File-format conventions

| Artifact | Format |
|---|---|
| Camera intrinsics | `.npz` with keys `K` (3,3), `dist` (5,), `image_size` (2,) |
| Annotations | COCO segmentation JSON (Roboflow export) |
| Model weights | PyTorch `state_dict` saved as `.pt` |
| Ground truth | CSV with `filename, width_mm, height_mm` |
| Per-run results | CSV with per-image preds, errors, confidence |
| Test metrics | JSON with `bbox_mAP`, `segm_mAP`, `precision`, `recall`, `f1`, etc. |
