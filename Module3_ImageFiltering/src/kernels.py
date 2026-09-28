"""
kernels.py
Builds the small grids of weights (the "kernels") that we use to blur an image.

A kernel is just a little square of numbers. To blur a pixel we take a weighted
average of that pixel and the pixels around it, and the kernel says how much
weight each neighbour gets. I set up two kinds here:

  box       every weight is the same, so it is a plain average of the neighbours
  gaussian  the weights follow a bell curve, so the middle pixel counts most and
            the effect fades out towards the edges (this looks smoother)

Every kernel is normalised, meaning all of its weights add up to 1. That keeps
the picture at the same overall brightness after blurring.
"""

from __future__ import annotations
import numpy as np


def box_kernel(size):
    """
    Make a size by size box (averaging) kernel. For size 3 this is a 3x3 grid
    where every entry is 1/9.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size has to be a positive odd number like 3, 5, 7")
    k = np.ones((size, size), dtype=np.float64)
    return k / k.sum()


def gaussian_kernel(size, sigma=None):
    """
    Make a size by size Gaussian kernel.

    The 2D Gaussian is exp(-(x^2 + y^2) / (2 * sigma^2)). We work out that value
    on a small grid centred at (0, 0) and then divide by the total so the weights
    add up to 1.

    sigma controls how wide the bell curve is. If you do not pass one, I use
    size/6, which is the usual rule of thumb (it puts about 3 sigma at the edge
    of the kernel).
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size has to be a positive odd number like 3, 5, 7")
    if sigma is None or sigma <= 0:
        sigma = size / 6.0

    half = size // 2
    # coordinate axis, e.g. for size 5 this is [-2, -1, 0, 1, 2]
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def make_kernel(kind, size, sigma=None):
    """Small helper so the rest of the app can just ask for 'box' or 'gaussian'."""
    kind = kind.lower()
    if kind == "box":
        return box_kernel(size)
    if kind == "gaussian":
        return gaussian_kernel(size, sigma)
    raise ValueError("kind should be 'box' or 'gaussian', got " + repr(kind))
