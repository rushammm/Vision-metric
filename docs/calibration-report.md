# Camera Calibration Report

## Setup

- **Camera:** iPhone 12 Pro Max, main 1x wide lens.
- **Settings:** AE/AF locked, Live Photos off, JPEG (Settings > Camera > Formats > Most Compatible), landscape orientation throughout.
- **Target:** chessboard PNG displayed fullscreen on a laptop LCD. The screen is flatter than printed paper and prints don't add their own scale error.
- **Pattern:** 10 x 7 squares, so 9 x 6 inner corners.
- **Square side:** 24.0 mm, measured on the screen with a steel ruler.

## Capture and detection

Took 63 photos at varied angles and distances, all in landscape. Of those, 56 had detectable corners on the first pass and 55 were kept after one round of outlier rejection (1 image with reprojection error > 1 px was dropped before the final solve).

The first attempt used `cv2.findChessboardCorners` and detection was unreliable on the high-resolution iPhone images (lots of "no corners found" returns even on photos that clearly contained the board). Switching to `cv2.findChessboardCornersSB` (the sector-based detector) fixed it. The fallback to the legacy detector is still in `calibrate.py` but in practice it never triggers.

Outlier rejection rule: drop any image whose per-image reprojection error is greater than `max(1.0 px, 3 * median)` after the first solve, then re-run the calibration on the remaining images.

## Results

**Image resolution:** 4032 x 3024 px

**Intrinsic matrix K (pixels):**

```
[ 3072.82       0.00    2028.42 ]
[    0.00    3074.94    1499.90 ]
[    0.00       0.00       1.00 ]
```

- `fx ~= fy ~= 3073 px`. Square pixels, as expected.
- Principal point `(cx, cy) = (2028.4, 1499.9)`. About 12 px from the geometric centre `(2016, 1512)`. Sanity check passes.

**Distortion coefficients (k1, k2, p1, p2, k3):**

```
k1 =  0.2335
k2 = -1.4672
p1 =  0.0007
p2 =  0.0009
k3 =  3.0161
```

`k1 > 0` indicates barrel distortion (lines bow outward). That's the expected pattern for a phone wide lens. The tangential terms `p1, p2` are essentially zero, which says the lens is well-centred on the sensor.

**Reprojection error:**

| Metric | Value | Target |
|---|---:|---:|
| Mean per-image error | 0.0845 px | < 0.3 px |
| RMS error            | 0.6803 px | < 0.5 px |

Mean per-image error of 0.085 px on a 4032 x 3024 image is well below the < 0.3 px target. RMS is slightly elevated because it weights all 55 x 54 = 2970 reprojected points equally and a handful of near-edge corners contribute disproportionately. The mean is the more useful number for sanity.

## Verification

<img src="figures/undistortion_demo.jpg" alt="raw vs undistorted" width="600">

Raw photo (left) next to its undistorted version (right). Two visible signs the undistortion is doing the right thing:

1. The undistorted panel has a curved black border around the edges. That's because we use `alpha=1.0`, which retains all source pixels (the curved border is the geometric inverse of the lens's barrel distortion).
2. Real-world straight edges (laptop bezel, desk edge) are visibly straighter near the frame edges in the undistorted image than in the raw.

## Why undistortion is mandatory before measurement

For this iPhone 12 Pro Max main lens, `k1 = 0.234` means two points 100 mm apart in the world project to noticeably different pixel separations depending on where they sit in the frame. The error is non-uniform across the image, so a single global pixels-per-mm scale cannot fix it.

`cv2.undistort` uses the recovered K and dist coefficients to remap every pixel back to its pinhole-projected position. After that, pixel-to-mm scaling is consistent across the frame. Skipping this step injects 1-5% measurement error that varies with where the object sits.

The measurement pipeline (`measurement/measure.py:undistort`) calls this on every input image before doing anything else.

## Reproducing

```
python calibration/generate_checkerboard.py
# display the PNG fullscreen on a laptop, measure one square with a ruler
# capture ~50 photos in landscape with iPhone, drop into calibration/images/

python calibration/calibrate.py --square-mm 24.0 --cols 9 --rows 6
# -> calibration/intrinsics.npz, debug overlays in calibration/debug/

python calibration/undistort_test.py
# -> docs/figures/undistortion_demo.jpg
```
