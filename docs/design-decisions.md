# Design Decisions

The non-obvious choices in this pipeline, what we picked, what we considered, and why we picked what we picked. Where a choice has a downside, it's called out.

---

## 1. Mask R-CNN over alternatives

**Picked:** `torchvision.models.detection.maskrcnn_resnet50_fpn`, COCO-pretrained, fine-tuned for 2 classes.

**Considered:**
- DeepLabV3+ / U-Net (semantic segmentation)
- SAM2 fine-tune
- Mask R-CNN with ResNet-101 backbone
- Detectron2

**Reason:** Mask R-CNN gives instance masks with confidence scores, which is what the measurement step actually needs (one ranked prediction per object). Semantic segmentation collapses all instances into one mask per class, which is fine here (one notebook per photo) but adds friction if we ever want multi-instance. SAM2 is heavyweight for no obvious accuracy gain on a flat rectangular object. Detectron2 adds an install/deploy footprint with no advantage at this scale.

ResNet-50 backbone is the well-trodden default for transfer learning; ResNet-101 didn't seem worth the extra training time on 48 images.

**Trade-off:** Mask R-CNN is heavy at inference (~1-2 sec on CPU). Acceptable here because measurement is a one-shot, click-driven workflow, not real-time.

---

## 2. Manual card click over auto card detection

**Picked:** user clicks 4 card corners on a downscaled preview window.

**Considered (and implemented as fallback):** classical contour detection - Canny edges + Otsu thresholding, filter contours by ID-1 aspect ratio (1.586), pick the highest-scoring rectangle.

**Reason:** auto-detection produces too many false positives on busy backgrounds. The notebook's own spine, table-edge shadows, and page edges in books all match the 1.586 aspect ratio. Several test photos picked the notebook itself as the "card" before any model inference ran. Diagnosing the failure mode is hard for a non-expert user.

The click-based flow takes ~5 seconds per image, removes a class of failures that's hard to diagnose, and the user is already in front of the screen anyway. For a measurement tool that already requires placing a card in frame, asking for four clicks is acceptable.

**Trade-off:** the pipeline is no longer fully automated. Auto-detection is still available behind `--auto-card` for scripted use.

---

## 3. Homography over global pixels-per-mm

**Picked:** compute a homography H from card pixels to mm on the table plane. Project mask contour points through H, then fit `minAreaRect` in mm space.

**Considered:** detect card, measure its long side in pixels, divide by 85.60 mm to get a global pixels-per-mm scale, then measure the notebook in pixels and convert.

**Reason:** the global-scale approach only works when the camera is exactly perpendicular to the table. Any tilt foreshortens the card and the notebook by different amounts depending on their position in the frame, and the conversion factor is wrong by 5-30 % for typical phone-handheld angles.

A homography handles tilt by construction: it maps any point on the table plane to its true mm coordinates, regardless of viewing angle.

**Trade-off:** requires the user to click 4 points instead of relying on automatic card detection. Already chose to do this anyway (decision #2), so the cost is zero on top.

**Limitation acknowledged:** a planar homography only works for points actually on the calibration plane. Our notebook is ~20 mm thick, so its top cover is off-plane. This produces the systematic over-estimate documented in `measurement-report.md`.

---

## 4. minAreaRect after homography, not before

**Picked:** project every mask contour pixel through H first, then fit `cv2.minAreaRect` in mm space.

**Considered:** fit `minAreaRect` on the pixel mask, then convert just the 4 rectangle corners to mm.

**Reason:** in a tilted-camera photo, a real-world rectangle projects as a trapezoid. Fitting `minAreaRect` to the trapezoid gives an axis-aligned bounding box of the trapezoid, which is **bigger** than the true rectangle and not even the same shape. Converting that wrong rectangle's corners to mm preserves the wrong measurement.

Doing the per-point projection first reverses the foreshortening: in mm-space, the notebook outline becomes a true rectangle, and `minAreaRect` then fits it tightly.

**Trade-off:** we project a few hundred contour points instead of 4. Negligible at this scale (microseconds).

---

## 5. Screen checkerboard over printed checkerboard

**Picked:** displayed the chessboard PNG fullscreen on a laptop LCD.

**Considered:** print the chessboard on paper and tape to a flat surface.

**Reason:**
- A laptop screen is **flatter** than printed-and-taped paper (paper bows under tape, prints bubble in humidity).
- We can measure one square directly on the screen with a ruler. Print-and-measure introduces print-quality and paper-shrinkage variations.
- The screen is high-contrast and well-lit, which `findChessboardCornersSB` likes.

**Trade-off:** the screen has a finite refresh rate and pixel grid. At very close distances you can see the pixel dots. We kept distance > 30 cm to avoid this.

---

## 6. `findChessboardCornersSB` over `findChessboardCorners`

**Picked:** OpenCV's sector-based corner detector (added in 4.0).

**Considered:** the legacy `findChessboardCorners`.

**Reason:** the legacy detector failed to find corners on roughly half of our 4032×3024 iPhone photos, even on photos that clearly contained the full board. SB ("sector based") detector, designed for high-resolution images and varied lighting, succeeded on 56/63.

**Trade-off:** SB is slightly slower per call (irrelevant - we run it 63 times once).

---

## 7. Two-pass calibration with outlier rejection

**Picked:** solve once on all detected images, drop any image with reprojection error > `max(1.0 px, 3 × median)`, re-solve.

**Considered:** single-pass, accept all detected images.

**Reason:** even with SB detector, occasional images have a few mis-detected corners that drag the reprojection error up. Dropping per-image outliers improved the mean per-image error from ~0.13 px to **0.085 px** with negligible loss of geometric coverage (1 image dropped, 55 retained).

---

## 8. Transfer learning over training from scratch

**Picked:** load COCO-pretrained Mask R-CNN, replace the box and mask predictor heads with 2-class versions, fine-tune all parameters for 10 epochs.

**Considered:** train from scratch with random initialization. Rejected immediately - 48 training images is far too few for that.

**Considered:** freeze the backbone, only train the new heads. Tried briefly during development; converged slower and to lower IoU. The full fine-tune at LR 5e-3 is fast enough (3 minutes on T4) that there's no reason to freeze.

---

## 9. SGD over Adam for fine-tuning

**Picked:** SGD with momentum 0.9, weight decay 5e-4, LR 5e-3, StepLR (step=4, gamma=0.5).

**Considered:** Adam / AdamW.

**Reason:** the torchvision detection examples and the original Mask R-CNN paper both use SGD-with-momentum at this scale. Adam tends to over-fit small datasets in detection tasks because it adapts learning rates per-parameter and the box/mask head gradients are noisy at small batch sizes. The default SGD recipe is well-trodden and we hit 0.96 val IoU in 10 epochs, so no reason to deviate.

---

## 10. Splitting outside Roboflow

**Picked:** export everything as `train/` from Roboflow, then split with `dataset/split.py` (random seed 42, image-level).

**Considered:** use Roboflow's built-in train/val/test split.

**Reason:** Roboflow's split is non-reproducible from code (you click a slider in the UI). We wanted the split to be a versioned artifact in the repo, so re-running `split.py` produces the exact same files. Also lets us inspect the split logic in code rather than trusting a UI.

**Trade-off:** Roboflow's per-class stratification is no longer applied. With one class, this is irrelevant.

---

## 11. Storing weights outside git

**Picked:** `.gitignore` excludes `*.pt`. Training is reproducible from `notebooks/train_colab.ipynb` in ~3 minutes on a free T4.

**Considered:** Git LFS for the 176 MB `best.pt`.

**Reason:** Git LFS adds account/setup friction for anyone cloning the repo; reproducing the training is simpler. Documented in README.

---

## 12. Same iPhone, same notebook

**Picked:** all calibration, training, and validation done on one iPhone 12 Pro Max and one notebook.

**Reason:** scoped to demonstrate the pipeline works end-to-end. Generalisation across devices and notebook styles would need fresh calibration per device and a substantially larger / more diverse dataset.

**Limitation acknowledged:** the model has only seen one notebook style; the pipeline would need re-calibration on any other phone.

---

## What was tried and discarded

- **Auto card detection in busy scenes** - too many false positives, kept as a `--auto-card` fallback only.
- **Freezing backbone during fine-tune** - slower convergence, lower IoU. Reverted to full fine-tune.
- **Single-pass calibration** - mean error 0.13 px. Two-pass with outlier rejection brought it to 0.085 px.
