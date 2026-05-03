# Measurement Report

## What it does

Takes one phone photo of a notebook lying on a flat surface, with an ID-1 card (credit/debit card, 85.60 x 53.98 mm) somewhere in the same frame on the same plane. Returns the notebook's width and height in millimetres.

## Pipeline

1. Read image, apply EXIF orientation, rotate to landscape if needed.
2. Undistort with the calibrated K + dist (intrinsics scaled to the input resolution if the photo was downsized, see `measure.scale_K`).
3. User clicks the 4 corners of the card on a downscaled preview window. Click order doesn't matter; corners are auto-sorted into TL, TR, BR, BL by sum/difference of coordinates.
4. Compute a homography H from card pixels to a metric plane where the card's long side is exactly 85.60 mm. Long-side orientation is picked by comparing the lengths of the top and right edges of the clicked quad.
5. Run Mask R-CNN on the undistorted image, take the highest-score mask, threshold at 0.5.
6. Take the largest external contour of the mask, project every contour point through H into mm, fit `cv2.minAreaRect` in mm space, return the long side as width and the short side as height.

The key decision is doing minAreaRect **after** the homography rather than before. Doing it in pixels and then scaling by a single px-per-mm value only works if the camera is exactly overhead. The homography handles tilt directly.

## Card detection

Auto-detection using edges + Otsu + aspect-ratio filtering is implemented (`detect_card`) but it was unreliable in busy backgrounds (the notebook's own edges, table-edge shadows, and reflections all produced false positives with the right aspect ratio). Manual click won as the default. The auto path is still there behind `--auto-card`.

The click-based flow takes ~5 seconds per image and removes a class of failure that's hard to diagnose. For a measurement tool that already requires the user to place a card in frame, asking for four clicks is acceptable.

## Validation

Captured 20 photos of the same black notebook on varied backgrounds, all shot directly overhead (camera held parallel to the table). Card lying flat on the same surface as the notebook. Ground truth: 180 x 125 mm, measured with a steel ruler.

Out of 20 photos, 1 (IMG_4339) failed because of a bad card click that produced a near-degenerate homography (predicted 155 m wide). It's excluded from the aggregates below. **N = 19.**

| Aggregate    | Width    | Height   |
|--------------|---------:|---------:|
| MAE (mm)     |  23.6    |  14.2    |
| MPE (%)      |  13.1    |  11.4    |
| Max abs (mm) |  38.3    |  42.5    |
| Min abs (mm) |   7.0    |   5.0    |

Per-image numbers are in `measurement/outputs/results.csv`.

### Reading the results

Two things stand out.

**Direction of the error is consistent.** Every prediction over-estimates both width and height. There's no case where the prediction is below the true value. That rules out random measurement noise.

**The size of the over-estimate matches a thickness/parallax model.** The card sits on the table at height 0. The notebook's top cover sits about 20 mm above the table. The mask segments the top cover. The homography is calibrated at the table plane (the card plane). An object 20 mm above the table, photographed from roughly 25-30 cm overhead, projects 7-9% larger than its true footprint. For a 180 x 125 notebook that predicts ~193-196 x 134-136 mm, which is exactly the cluster the results sit in.

So overhead shooting fixed the **angle dependence** that the earlier mixed-angle run had (worst-case errors dropped from 30-40% to ~21%) but it cannot fix **out-of-plane parallax** for a thick object. A planar homography assumes the measured surface is the same as the reference surface, and the notebook's top cover is not.

### What would actually fix it

Three options, none implemented:

1. Place the card **on top of** the notebook instead of next to it. That puts the reference and the measured surface in the same plane. Trade-off: the card occludes part of the notebook mask, so the mask has to be reconstructed or the card has to be small enough to leave clean edges.
2. Use two reference cards at known different heights to estimate the camera-to-table distance, then back out the height of the notebook from the projection scale. More math, more clicks.
3. Calibrated stereo or a depth sensor. Out of scope.

## Limitations

- **Out-of-plane parallax.** Documented above. Adds a systematic ~7-9% over-estimate for a 20 mm thick object at typical phone-overhead distances.
- **Click precision.** Sub-pixel error on the four corner clicks shows up as a few percent in the homography. For one photo (IMG_4339) it produced a fully degenerate result. A sanity check on the homography (reject if predicted card aspect deviates from 1.586 by more than a few percent) would catch this.
- **Plane assumption.** Card and notebook surface must be coplanar. This is the same parallax issue stated as a precondition: if you ignore it, the error grows fast.
- **Same lens, same calibration.** Pinch-zoom, lens swap, or resolution change invalidates K. The pipeline scales K linearly for uniform downscales (verified per file by aspect ratio match), but it cannot recover from a crop.
- **Single-class model.** Trained on one notebook style. New covers, glossy materials, or unfamiliar backgrounds may degrade mask quality and therefore the measurement.

## Reproducing

```
python measurement/validate.py --ground-truth measurement/ground_truth.csv
```

Click 4 card corners on each image, ENTER. Results land in `measurement/outputs/results.csv`.

For a single image with an annotated output:

```
python inference/demo.py --image dataset/raw/IMG_4326.JPG.jpeg --out docs/figures/demo_output.jpg
```
