# Dataset Card

## Object

- **Class:** `notebook` (single foreground class).
- **Real-world dimensions:** 180 x 125 mm, ~20 mm thick. Black leather-style cover, no text or pattern on the front.
- **Why this object:** flat rectangular profile, matte cover so no specular highlights to confuse the mask, and easy to measure with a ruler for ground truth.

## Capture

- **Device:** iPhone 12 Pro Max, main 1x wide lens (same lens used for calibration).
- **Settings:** AE/AF locked on the notebook before each shot, JPEG (Settings > Camera > Formats > Most Compatible), Live Photos off, landscape orientation.
- **Image count:** 68 raw photos.
- **Variation:**
  - Backgrounds: wood desk, fabric, tile, patterned cloth, plain paper.
  - Lighting: daylight by a window, warm indoor lamp, mixed.
  - Angles: roughly overhead through to ~30 degrees of tilt.
  - Distances: about 25-50 cm from lens to notebook.
- **Reference object in frame:** present in the photos used for measurement validation, not required for training/labelling photos.

## Contact sheet

All 68 raw photos:

<img src="figures/contact_sheet.jpg" alt="dataset contact sheet" width="450">

## Sources and gotchas

The 68 raw photos came from two batches taken on different days and transferred via WhatsApp at different times. That introduced two annoying inconsistencies:

1. **Mixed codecs.** Some files arrived as real JPEGs, some as HEIC files renamed `.jpeg` (iOS does this when WhatsApp's source is "Most Compatible" but the original was HEIC). All reads go through Pillow + `pillow_heif` to handle both.
2. **Mixed resolutions.** WhatsApp transcoded some photos uniformly (no crop) to 2181 x 1636, others stayed at the native 4032 x 3024. Aspect ratio is preserved in both, so K is rescaled linearly per resolution (`scale_K`).

Both are handled in `dataset/normalize_and_undistort.py`, which:
- Reads any of {JPEG, PNG, HEIC, HEIF}.
- Applies EXIF orientation.
- Rotates portrait to landscape (so the calibrated K orientation applies).
- Rejects images with mismatched aspect ratio (signals a crop, not a clean downscale).
- Scales K per resolution and undistorts.

Output: `dataset/undistorted/`. These are what got labelled.

## Labelling

- **Tool:** Roboflow, Smart Polygon (SAM-assisted) for the initial mask, then manual cleanup of corner leakage.
- **Annotation type:** polygon segmentation, exported in COCO segmentation format, single class `notebook` (Roboflow assigns class ID 1; class 0 is background by convention).
- **QC:** every mask checked at full resolution; soft edges on the spine were tightened by hand. About 12 of 68 images needed manual touch-up.

A few HEIC files made it through Roboflow's pipeline with a `.jpeg` extension but were still HEIC bytes inside (Roboflow renamed but didn't transcode). Those needed in-place re-encoding to real JPEG inside `dataset/splits/` before training would read them with `cv2.imread`.

## Splits

Roboflow exported everything as `train/` only, so we split outside of Roboflow with `dataset/split.py` (random seed 42, image-level so there's no augmentation leakage between splits).

| Split | Count | Fraction |
|------:|------:|---------:|
| train |    48 |   0.706  |
| val   |    14 |   0.206  |
| test  |     6 |   0.088  |

## Resizing for Colab upload

Full-resolution undistorted images are large (~3-5 MB each, 68 of them). For the Colab training notebook the splits are downscaled with `dataset/prep_for_colab.py` (longest side capped at 1280 px, JPEG quality 85). Output goes to `dataset/splits_small/`, total ~20 MB, which uploads in seconds.

The original `dataset/splits/` is kept for any later experiments at native resolution.

## Limitations

- **68 photos vs the spec's 70+.** Two short. We considered re-shooting to fill the gap but decided the time was better spent on the validation re-shoot. The training result (val IoU 0.96) is strong enough that a couple more images are unlikely to move the headline number, but it's a known shortfall.
- **One notebook.** Same physical object across all 68 photos. The model has not seen any other notebook style, so generalisation is unverified.
- **No held-back unseen-instance test.** Splits are by photo, not by object instance. With only one object the model can't really demonstrate generalisation across instances.

## Files

```
dataset/
  raw/                    # original phone photos (mixed JPEG / HEIC-as-jpeg)
  undistorted/            # after EXIF + rotate-to-landscape + cv2.undistort
  splits/                 # Roboflow export, then split.py -> train/val/test
    train/_annotations.coco.json + 48 images
    val/_annotations.coco.json   + 14 images
    test/_annotations.coco.json  +  6 images
  splits_small/           # same as splits/, longest side <= 1280 px (for Colab upload)
  notebook-segmentation.coco-segmentation/   # raw Roboflow export (kept as a backup)
  normalize_and_undistort.py
  split.py
  prep_for_colab.py
```
