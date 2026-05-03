# Generate a checkerboard PNG. Display fullscreen, measure one square with
# a ruler, pass that mm value to calibrate.py.

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_checkerboard(cols, rows, square_px, out_path):
    # cols/rows are NUMBER OF SQUARES. Inner corners (for OpenCV) = (cols-1, rows-1).
    board = np.ones((rows * square_px, cols * square_px), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                board[r * square_px:(r + 1) * square_px,
                      c * square_px:(c + 1) * square_px] = 0

    border = square_px // 2
    h, w = board.shape
    bordered = np.ones((h + 2 * border, w + 2 * border), dtype=np.uint8) * 255
    bordered[border:border + h, border:border + w] = board

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bordered).save(out_path)
    print(f"saved {out_path}")
    print(f"squares: {cols} x {rows}, inner corners: {cols - 1} x {rows - 1}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=10)
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--square-px", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("calibration/checkerboard.png"))
    args = p.parse_args()
    make_checkerboard(args.cols, args.rows, args.square_px, args.out)
