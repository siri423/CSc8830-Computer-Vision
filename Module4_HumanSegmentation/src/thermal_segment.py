"""
Q2: find the boundary of a person in a thermal (infrared) image using only
classical OpenCV. No deep learning or machine learning.

Run:
    python -m src.thermal_segment --image data/person_thermal.png --out outputs

Why thermal is different from RGB:
  A warm body radiates more than the cool ground, so a person shows up as a
  bright, compact, roughly vertical blob. Colour models are not needed. The
  catch is that other things are hot too (sunlit walls, lit windows), so the
  brightest pixel is not always the person.

We use a two-stage, coarse-to-fine method:

  Stage 1, locate the person:
    blur, threshold high (max of Otsu and a high percentile) so only the hottest
    regions remain, clean up with morphology, then score each hot blob on how
    human it looks using shape and context only:
      aspect ratio near 2:1 (upright person, not a thin window),
      a ground-plane prior (people stand low in the frame, not up on a wall),
      isolation (a person is ringed by cool ground; a window sits in a warm wall),
      size and solidity.
    The best-scoring blob is the person.

  Stage 2, refine the boundary:
    crop a box around the located person (extended downward for the dimmer legs)
    and run a local Otsu inside it. The box leaves out the far-off hot windows,
    so a lower local threshold safely recovers the whole body. Fill holes and
    trace the contour.
"""

import argparse
import os
import cv2
import numpy as np

from .utils import (
    imread_gray, imread_color, fill_holes, keep_largest_component,
    draw_boundary, shade_mask, cutout, stack_h, OURS_BGR,
)


def _score_blob(stats, centroid, labels, gray, idx, kernel):
    """
    Score one hot blob on how human it looks (higher is better). Returns -1 to
    reject a blob outright. Uses geometry and local contrast only.
    """
    x, y, w, h, area = (stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP],
                        stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT],
                        stats[idx, cv2.CC_STAT_AREA])
    cy = centroid[idx][1]
    H = gray.shape[0]
    aspect = h / (w + 1e-6)
    extent = area / (w * h + 1e-6)
    rel_h = h / H

    if area < 30 or h < 12 or aspect < 1.1 or aspect > 5.0 or rel_h > 0.9:
        return -1.0

    # isolation: how much brighter the blob is than a ring of pixels around it
    comp = (labels == idx).astype(np.uint8)
    ring = (cv2.dilate(comp, kernel, iterations=3) - comp) > 0
    surround = gray[ring].mean() if ring.sum() else 0.0
    isolation = (gray[comp > 0].mean() - surround) / 255.0

    aspect_s = max(0.0, 1.0 - abs(aspect - 2.2) / 2.2)    # peaks at a human ~2.2:1
    pos_s = min(max((cy / H - 0.2) / 0.3, 0.0), 1.0)      # ground-plane prior
    size_s = min(rel_h, 0.5) / 0.5

    return (aspect_s * 1.3 + pos_s * 1.3 + size_s * 0.6
            + max(isolation, 0.0) * 1.0 + extent * 0.4)


def segment_thermal(gray, blur=3, morph_kernel=5, hot_percentile=96):
    """
    Segment the person from a thermal grayscale image, coarse to fine.

    gray:           thermal image, single channel uint8
    blur:           Gaussian blur size (odd) to reduce sensor noise
    morph_kernel:   structuring-element size for clean-up
    hot_percentile: stage-1 threshold is max(Otsu, this percentile)
    Returns a binary mask (uint8, 0 or 255).
    """
    H, W = gray.shape
    gb = cv2.GaussianBlur(gray, (blur, blur), 0) if blur >= 3 else gray
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    # Stage 1: locate the person.
    otsu, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), float(np.percentile(gb, hot_percentile)))
    binary = (gb >= thr).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_idx, best_score = -1, 0.0
    for i in range(1, num):
        s = _score_blob(stats, cent, labels, gb, i, k)
        if s > best_score:
            best_score, best_idx = s, i

    if best_idx == -1:
        # Nothing scored as a person, so fall back to the largest hot blob.
        mask = np.zeros_like(binary)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask[labels == largest] = 255
        return fill_holes(mask)

    # Stage 2: refine the boundary with a local Otsu around the located person.
    x, y, w, h = (stats[best_idx, cv2.CC_STAT_LEFT], stats[best_idx, cv2.CC_STAT_TOP],
                  stats[best_idx, cv2.CC_STAT_WIDTH], stats[best_idx, cv2.CC_STAT_HEIGHT])
    mx = int(0.9 * w)
    m_top, m_bot = int(0.3 * h), int(1.1 * h)     # extend down to catch the legs
    x0, x1 = max(0, x - mx), min(W, x + w + mx)
    y0, y1 = max(0, y - m_top), min(H, y + h + m_bot)

    roi = gb[y0:y1, x0:x1]
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, k, iterations=2)
    roi_bin = keep_largest_component(roi_bin)

    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = roi_bin
    mask = fill_holes(mask)
    return mask


def run(image_path, out_dir="outputs", hot_percentile=96):
    """Segment one thermal image and write the result figures. Returns the mask."""
    os.makedirs(out_dir, exist_ok=True)
    gray = imread_gray(image_path)
    bgr = imread_color(image_path)
    mask = segment_thermal(gray, hot_percentile=hot_percentile)

    boundary = draw_boundary(bgr, mask, OURS_BGR, 2)
    shaded = shade_mask(bgr, mask, OURS_BGR, 0.45)
    cut = cutout(bgr, mask)

    cv2.imwrite(os.path.join(out_dir, "thermal_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "thermal_boundary.png"), boundary)
    cv2.imwrite(os.path.join(out_dir, "thermal_cutout.png"), cut)
    cv2.imwrite(os.path.join(out_dir, "thermal_panel.png"),
                stack_h([bgr, shaded, boundary, cut], height=360))

    pct = 100 * cv2.countNonZero(mask) / mask.size
    print(f"[thermal] foreground {int(cv2.countNonZero(mask))} px ({pct:.1f}% of image)")
    print(f"[thermal] wrote results to {out_dir}/thermal_*.png")
    return mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Classical thermal human boundary.")
    ap.add_argument("--image", default="data/person_thermal.png")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--percentile", type=float, default=96,
                    help="stage-1 threshold is max(Otsu, this percentile)")
    args = ap.parse_args()
    run(args.image, args.out, hot_percentile=args.percentile)
