# Setup

## Prerequisites

- Python 3.11 or 3.12 recommended (see version note below)
- Git
- ~3 GB free disk for dataset + model weights
- An iPhone 12 Pro Max if you want to use the existing calibration as-is, otherwise any phone (re-run calibration if you switch device)

> **Python version note.** PyTorch wheels target Python 3.9-3.12. If you have 3.13 or 3.14, calibration and measurement run fine (they only need `opencv-contrib-python`, `numpy`, `pillow`, `pillow_heif`), but training needs a separate venv on 3.11 or 3.12. The local dev here used 3.14 + torch 2.11 CPU for everything except training, which ran on Colab.

## Install

```
git clone <this-repo-url> xis-assessment
cd xis-assessment

python -m venv .venv
.venv\Scripts\activate                   # Windows PowerShell
# source .venv/bin/activate               # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` fails to install for your Python version, install Python 3.11 from python.org and recreate the venv.

## iPhone capture settings (one-time)

`K` is only meaningful if image formation stays the same across all photos. Lock these once and don't change them between calibration and measurement:

1. **Settings > Camera > Formats > Most Compatible** (saves JPEG, not HEIC).
2. Use the **1x main wide lens only**. No pinch zoom (that crops digitally or switches lens).
3. Same orientation for calibration and measurement (landscape was used here).
4. **Lock AE/AF**: long-press on the subject in the Camera app until "AE/AF LOCK" shows.
5. Live Photos off.

If you change resolution, lens, or any of the above between calibration and measurement, the calibration is invalid.

## Run order

The full pipeline, in the order it was actually run:

```
# 1. Generate the checkerboard, display it fullscreen on a laptop,
#    measure one square with a ruler.
python calibration/generate_checkerboard.py
# -> calibration/checkerboard.png

# 2. Capture ~50 photos of the screen at varied angles/distances,
#    drop into calibration/images/, then:
python calibration/calibrate.py --square-mm 24.0 --cols 9 --rows 6
# -> calibration/intrinsics.npz, debug overlays in calibration/debug/

# 3. Verify the undistortion looks right.
python calibration/undistort_test.py
# -> docs/figures/undistortion_demo.jpg

# 4. Capture notebook photos for training/labelling, drop into dataset/raw/,
#    then normalise and undistort.
python dataset/normalize_and_undistort.py
# -> dataset/undistorted/

# 5. Label in Roboflow (Smart Polygon, single class `notebook`),
#    export COCO segmentation, drop into dataset/, then split:
python dataset/split.py
# -> dataset/splits/{train,val,test}/

# 6. Optional: shrink for Colab upload.
python dataset/prep_for_colab.py
# -> dataset/splits_small/

# 7. Train. Locally:
python models/train.py --data-dir dataset/splits --epochs 10 --batch-size 2
# Or on Colab via notebooks/train_colab.ipynb (recommended).
# -> models/checkpoints/best.pt, models/checkpoints/training_log.csv

# 8. Validate measurement accuracy on a list of ground-truth photos.
python measurement/validate.py --ground-truth measurement/ground_truth.csv
# Click 4 card corners on each image, ENTER. -> measurement/outputs/results.csv

# 9. End-to-end demo on a single image.
python inference/demo.py --image dataset/raw/IMG_4326.JPG.jpeg --out docs/figures/demo_output.jpg
# Click 4 card corners, ENTER. -> annotated JPEG with width/height/confidence
```

Steps 1-3 only need to be re-run if the camera changes. Steps 4-7 only if the dataset or model changes. Steps 8-9 are the everyday measurement workflow.
