"""
Compare our classical masks against the SAM reference masks.

SAM is used only as the reference the assignment asks us to compare against. Our
own segmentation (rgb_segment.py, thermal_segment.py) never uses machine
learning.

Run after the SAM masks are saved in sam2/:
    python -m src.compare

Files it expects:
    outputs/rgb_mask.png       from src.rgb_segment
    outputs/thermal_mask.png   from src.thermal_segment
    sam2/rgb_sam.png           SAM mask for the RGB image
    sam2/thermal_sam.png       SAM mask for the thermal image

Metrics:
    IoU (Jaccard)     overlap over union, 1.0 is perfect
    Dice (F1)         2*overlap / (sizeA + sizeB), 1.0 is perfect
    Boundary F-score  how well the two outlines agree within a few pixels, which
                      is the "exact boundary" quality the task cares about
"""

import os
import json
import cv2
import numpy as np

from .utils import imread_color, draw_boundary, stack_h, OURS_BGR, SAM_BGR


def _binarize(mask):
    """Turn any mask image into a boolean array (foreground True)."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 127


def iou(a, b):
    """Intersection over union of two boolean masks."""
    a, b = _binarize(a), _binarize(b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def dice(a, b):
    """Dice coefficient (F1) of two boolean masks."""
    a, b = _binarize(a), _binarize(b)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2.0 * inter / float(s) if s else 0.0


def boundary_f_score(a, b, tol=2):
    """
    Boundary F-score. Precision and recall between the two outlines, where a
    boundary pixel counts as matched if the other outline passes within tol
    pixels. This rewards getting the edge right, which pixel-count metrics miss.
    """
    a, b = _binarize(a).astype(np.uint8), _binarize(b).astype(np.uint8)

    def edges(m):
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    ea, eb = edges(a), edges(b)
    da = cv2.distanceTransform(1 - eb, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1 - ea, cv2.DIST_L2, 3)
    ea_pts, eb_pts = ea > 0, eb > 0
    if ea_pts.sum() == 0 or eb_pts.sum() == 0:
        return 0.0
    precision = (da[ea_pts] <= tol).mean()
    recall = (db[eb_pts] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compare_pair(name, image_path, ours_path, sam_path, out_dir="outputs"):
    """Compute metrics for one image and write the outline comparison figure."""
    ours = cv2.imread(ours_path, cv2.IMREAD_GRAYSCALE)
    sam = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
    if ours is None or sam is None:
        print(f"[compare] skipping {name}: missing mask "
              f"(ours={ours is not None}, sam={sam is not None})")
        return None

    # The SAM mask may be a different size, so match it to ours.
    if sam.shape != ours.shape:
        sam = cv2.resize(sam, (ours.shape[1], ours.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    result = {
        "image": name,
        "IoU": round(float(iou(ours, sam)), 4),
        "Dice": round(float(dice(ours, sam)), 4),
        "BoundaryF@2px": round(float(boundary_f_score(ours, sam, tol=2)), 4),
    }

    # Figure: our outline (blue) and the SAM outline (green) on the image.
    bgr = imread_color(image_path)
    both = draw_boundary(bgr, ours > 127, OURS_BGR, 2)
    both = draw_boundary(both, sam > 127, SAM_BGR, 2)
    panel = stack_h([bgr, both], height=380)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{name}_compare.png"), panel)
    print(f"[compare] {name}: {result}")
    return result


def run(out_dir="outputs", sam_dir="sam2"):
    """Run the comparison for both images and save the metrics."""
    results = []
    r = compare_pair("rgb", "data/person_rgb.png",
                     os.path.join(out_dir, "rgb_mask.png"),
                     os.path.join(sam_dir, "rgb_sam.png"), out_dir)
    if r:
        results.append(r)
    r = compare_pair("thermal", "data/person_thermal.png",
                     os.path.join(out_dir, "thermal_mask.png"),
                     os.path.join(sam_dir, "thermal_sam.png"), out_dir)
    if r:
        results.append(r)

    if results:
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[compare] wrote {out_dir}/metrics.json")
    else:
        print("[compare] no SAM masks found; add them to sam2/ and run again.")
    return results


if __name__ == "__main__":
    run()
