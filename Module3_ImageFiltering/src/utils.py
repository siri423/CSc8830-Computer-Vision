"""
utils.py
Helper functions for loading images, saving them, and running a filter on a
color image one channel at a time. I kept the dependencies light on purpose:
only NumPy and Pillow.
"""

from __future__ import annotations
import numpy as np
from PIL import Image


def load_image(path, grayscale=True):
    """
    Load an image from disk and return it as a float array with values from
    0 to 1. If grayscale is True the image comes back as a single (H, W) plane,
    otherwise as (H, W, 3) with the red, green and blue channels.
    """
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    # divide by 255 so the pixel values sit between 0 and 1 instead of 0 and 255
    return np.asarray(img, dtype=np.float64) / 255.0


def to_uint8(arr):
    """
    Turn a float image (values roughly 0 to 1) back into the 0 to 255 integer
    format that images are normally stored in. Anything outside 0..1 is clipped
    first so it does not wrap around.
    """
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_image(arr, path):
    """Save a float image (0 to 1) to disk as a regular 8 bit image."""
    Image.fromarray(to_uint8(arr)).save(path)


def apply_per_channel(func, image, *args, **kwargs):
    """
    Run a 2D filtering function on an image that might be gray or color.

    A gray image is just (H, W), so we filter it directly. A color image is
    (H, W, 3), so we run the same filter on the red, green and blue planes
    separately and then stack them back together. That way the blurring code
    does not need to know or care whether the image has color.
    """
    if image.ndim == 2:
        return func(image, *args, **kwargs)
    channels = []
    for c in range(image.shape[2]):
        channels.append(func(image[:, :, c], *args, **kwargs))
    return np.stack(channels, axis=2)
