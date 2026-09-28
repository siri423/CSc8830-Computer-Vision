"""
Shared helpers: reading images, drawing the segmentation boundary, and laying
images out for the figures. Nothing here does any learning; it is image I/O and
drawing only.

The other scripts import these functions.
"""

import cv2
import numpy as np

# Colours used across the project (BGR order, since OpenCV uses BGR).
# We deliberately avoid red so the overlays read clearly on both the RGB and
# the thermal images.
OURS_BGR = (255, 144, 30)    # dodger blue, used for our classical result
SAM_BGR = (113, 204, 46)     # emerald green, used for the SAM reference


def imread_color(path):
    """Read an image as a 3-channel BGR image."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imread_gray(path):
    """Read an image as single-channel grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_rgb(bgr):
    """Convert BGR (OpenCV order) to RGB (for matplotlib or Streamlit)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def keep_largest_component(mask):
    """
    Keep only the largest white blob in a binary mask.

    A person is one connected object, so once we have a binary mask we drop the
    small stray blobs and keep the biggest one. Input and output are uint8 (0 or
    255).
    """
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return (mask * 255).astype(np.uint8)
    # index 0 is the background, so search components 1..num-1 for the largest
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.where(labels == largest, 255, 0).astype(np.uint8)
    return out


def fill_holes(mask):
    """
    Fill interior holes in a binary mask.

    Flood-fill the background starting from a corner, invert that, and OR it back
    in. Whatever the flood could not reach from outside is an interior hole.
    """
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)   # floodFill needs a 1px border
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def mask_to_contour(mask):
    """Return the outer contour of the largest blob in a mask, or None."""
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_boundary(bgr, mask, color=OURS_BGR, thickness=2):
    """
    Draw the outline of the mask on top of the image and return a new image.
    color is in BGR order.
    """
    out = bgr.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def shade_mask(bgr, mask, color=OURS_BGR, alpha=0.45):
    """Tint the foreground region with a translucent colour, for figures."""
    out = bgr.copy()
    layer = np.zeros_like(bgr)
    layer[mask > 0] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0.0)


def cutout(bgr, mask, bg=0):
    """Return the image with the background removed (kept where mask > 0)."""
    out = np.full_like(bgr, bg)
    out[mask > 0] = bgr[mask > 0]
    return out


def stack_h(images, height=360, pad=8, bg=30):
    """
    Resize a list of images to a common height and place them in one row.
    Used to build the before and after figures.
    """
    resized = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        scale = height / h
        resized.append(cv2.resize(im, (int(w * scale), height)))
    total_w = sum(im.shape[1] for im in resized) + pad * (len(resized) + 1)
    canvas = np.full((height + 2 * pad, total_w, 3), bg, np.uint8)
    x = pad
    for im in resized:
        canvas[pad:pad + height, x:x + im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas
