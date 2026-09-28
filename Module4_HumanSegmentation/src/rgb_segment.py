"""
Q1: find the boundary of a person in a regular colour (RGB) image using only
classical OpenCV. No deep learning or machine learning.

Run:
    python -m src.rgb_segment --image data/person_rgb.png --out outputs

Approach:
  We use GrabCut. GrabCut separates foreground from background by fitting colour
  Gaussian mixtures and cutting the lowest-cost boundary between them. It learns
  those colours from this one image, so it is not a trained model.

  1. Seed GrabCut automatically with a trimap (no clicking).
  2. Run GrabCut.
  3. Clean the mask: morphology, keep the largest blob, fill holes.
  4. Trace the outer contour, which is the boundary we report.
"""

import argparse
import os
import cv2
import numpy as np

from .utils import (
    imread_color, keep_largest_component, fill_holes,
    draw_boundary, shade_mask, cutout, stack_h, OURS_BGR,
)


def _build_trimap(h, w, border=0.03, col=(0.28, 0.72), col_y=(0.05, 0.98),
                  core=(0.40, 0.60), core_y=(0.35, 0.85)):
    """
    Build an automatic GrabCut seed for a roughly centred subject.

    We mark four things for GrabCut without any clicking:
      a thin outer border      -> sure background
      a central column         -> probable foreground (head down to feet)
      a core torso block       -> sure foreground
      everything else          -> probable background

    The sure-foreground core is what keeps parts whose colour matches the
    background (hair against a beige wall) from being removed.
    """
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    b = int(border * min(h, w))
    m[:b, :] = cv2.GC_BGD
    m[-b:, :] = cv2.GC_BGD
    m[:, :b] = cv2.GC_BGD
    m[:, -b:] = cv2.GC_BGD
    m[int(col_y[0] * h):int(col_y[1] * h),
      int(col[0] * w):int(col[1] * w)] = cv2.GC_PR_FGD
    m[int(core_y[0] * h):int(core_y[1] * h),
      int(core[0] * w):int(core[1] * w)] = cv2.GC_FGD
    return m


def segment_rgb(bgr, grabcut_iters=8, morph_kernel=5):
    """
    Segment the main person or foreground object with GrabCut.

    bgr:           input image (BGR uint8)
    grabcut_iters: GrabCut iterations (more is tighter but slower)
    morph_kernel:  size of the clean-up structuring element
    Returns a binary mask (uint8, 0 or 255).
    """
    h, w = bgr.shape[:2]

    # 1. Automatic trimap seed.
    mask = _build_trimap(h, w)

    # 2. GrabCut with the seed. bgd_model/fgd_model are working buffers OpenCV
    #    fills in.
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd_model, fgd_model,
                grabcut_iters, cv2.GC_INIT_WITH_MASK)

    # GrabCut marks pixels 0/2 as background and 1/3 as foreground.
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 3. Clean up.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)    # remove speckle
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)   # close small gaps
    fg = keep_largest_component(fg)
    fg = fill_holes(fg)
    return fg


def run(image_path, out_dir="outputs"):
    """Segment one RGB image and write the result figures. Returns the mask."""
    os.makedirs(out_dir, exist_ok=True)
    bgr = imread_color(image_path)
    mask = segment_rgb(bgr)

    boundary = draw_boundary(bgr, mask, OURS_BGR, 2)
    shaded = shade_mask(bgr, mask, OURS_BGR, 0.45)
    cut = cutout(bgr, mask)

    cv2.imwrite(os.path.join(out_dir, "rgb_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "rgb_boundary.png"), boundary)
    cv2.imwrite(os.path.join(out_dir, "rgb_cutout.png"), cut)
    cv2.imwrite(os.path.join(out_dir, "rgb_panel.png"),
                stack_h([bgr, shaded, boundary, cut], height=360))

    pct = 100 * cv2.countNonZero(mask) / mask.size
    print(f"[rgb] foreground {int(cv2.countNonZero(mask))} px ({pct:.1f}% of image)")
    print(f"[rgb] wrote results to {out_dir}/rgb_*.png")
    return mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Classical RGB human boundary (GrabCut).")
    ap.add_argument("--image", default="data/person_rgb.png")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    run(args.image, args.out)
