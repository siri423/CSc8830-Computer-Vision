"""
frequency.py
Blurring in the frequency (Fourier) domain.

The Convolution Theorem says that convolving in space is the same as multiplying
in frequency:

    f * g  in space   <=>   F . G  in frequency   (a plain element-wise product)

So instead of sliding a kernel over the image, we can:
  1. take the Fourier transform of the image   -> F
  2. take the Fourier transform of the kernel   -> G
  3. multiply them together                      -> F . G
  4. take the inverse Fourier transform          -> the blurred image

I wrote this so its result lines up with the hand written spatial convolution in
spatial.py (with zero padding) down to tiny rounding error. That match is what
proves the theorem in the app.

One detail that matters: the plain FFT gives a circular convolution, where the
kernel wraps around the edges of the image. To get an ordinary linear
convolution instead, I zero pad both arrays to size (H+kh-1, W+kw-1) before
transforming, and then crop the middle back out. The FFT itself is the one piece
I use NumPy for (np.fft); everything around it is done here.
"""

from __future__ import annotations
import numpy as np


def blur_frequency(image, kernel):
    """
    Blur a grayscale image by multiplying in the Fourier domain. The result is
    the same (H, W) image you would get from the zero padded spatial convolution.
    """
    H, W = image.shape
    kh, kw = kernel.shape

    # size we need for a linear convolution, so the kernel does not wrap around
    full_h, full_w = H + kh - 1, W + kw - 1

    # steps 1 and 2: transform the image and the kernel, both padded to full size
    F = np.fft.fft2(image, s=(full_h, full_w))
    G = np.fft.fft2(kernel, s=(full_h, full_w))

    # step 3: multiply in frequency, which is the same as convolving in space
    product = F * G

    # step 4: go back to an image and keep the real part (the imaginary part is ~0)
    full = np.real(np.fft.ifft2(product))

    # crop the middle so the output is the same size as the spatial version
    ph, pw = kh // 2, kw // 2
    return full[ph:ph + H, pw:pw + W]


def magnitude_spectrum(image):
    """
    Return a log magnitude spectrum of an image, shifted so the zero frequency
    sits in the middle. This is only for showing what the transform looks like.
    The output is scaled to 0..1 so it displays nicely.
    """
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F_shift))
    mag -= mag.min()
    if mag.max() > 0:
        mag /= mag.max()
    return mag


def kernel_frequency_response(kernel, shape):
    """
    Show what the kernel does in frequency (its transfer function) at a given
    image size. A blur kernel is a low pass filter, so this picture is bright in
    the middle (low frequencies kept) and dark near the edges (high frequencies,
    the fine detail, taken away). Scaled to 0..1 for display.
    """
    G = np.fft.fft2(kernel, s=shape)
    G_shift = np.fft.fftshift(G)
    mag = np.abs(G_shift)
    if mag.max() > 0:
        mag /= mag.max()
    return mag
