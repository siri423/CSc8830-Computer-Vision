"""
spatial.py
Blurring in the spatial domain, written out by hand. I am not calling any
ready made blur function here (no cv2.GaussianBlur and no scipy). This is the
"filtering approach" the assignment asks for.

Blurring is a 2D convolution of the image with a blur kernel. For each output
pixel you line the kernel up on the image, multiply the overlapping values, and
add them up.

There are two versions below:

  convolve2d_naive  the plain version with nested loops. It is slow but it makes
                    the definition obvious, which is nice for a small example.
  convolve2d        a faster version that gives the exact same answer. Instead of
                    looping over every pixel it loops over the handful of kernel
                    positions, which is much quicker on a real photo.

About the edges. When the kernel hangs off the side of the image we have to
decide what the missing pixels are:
  "zero"     treat anything outside as 0. This gives an ordinary linear
             convolution, which is exactly what the FFT method computes, so I use
             this when I want the two methods to match.
  "reflect"  mirror the edge pixels. This looks nicer for a plain visual blur
             because it does not darken the border.
"""

from __future__ import annotations
import numpy as np


def pad_image(image, pad_h, pad_w, mode="zero"):
    """Add a border of width (pad_h, pad_w) around the image."""
    if mode == "zero":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    if mode == "reflect":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    raise ValueError("mode should be 'zero' or 'reflect'")


def convolve2d_naive(image, kernel, mode="zero"):
    """
    The straightforward version of 2D convolution with nested loops. It is slow
    on big images but easy to read. The kernel is flipped first so that this is
    real convolution and not correlation (for a symmetric blur kernel the two
    are the same anyway).
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    kflip = kernel[::-1, ::-1]
    padded = pad_image(image, ph, pw, mode)
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            region = padded[y:y + kh, x:x + kw]
            out[y, x] = np.sum(region * kflip)
    return out


def convolve2d(image, kernel, mode="zero"):
    """
    The fast version. It gives the same result as convolve2d_naive.

    The trick: rather than looping over every pixel, loop over the kernel
    positions. For each weight in the kernel, take the whole padded image,
    shift it, multiply by that one weight, and add it on. A k by k kernel then
    only needs k*k array operations no matter how large the image is.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    kflip = kernel[::-1, ::-1]
    padded = pad_image(image, ph, pw, mode)
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            # padded[i:i+H, j:j+W] is the image shifted by (i, j)
            out += kflip[i, j] * padded[i:i + H, j:j + W]
    return out


def blur_spatial(image, kernel, mode="zero"):
    """Blur a grayscale image by convolving it with the kernel in the spatial domain."""
    return convolve2d(image, kernel, mode=mode)
