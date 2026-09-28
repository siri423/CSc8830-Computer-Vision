"""
Q3 made concrete: edge detection and region segmentation in the Fourier
(frequency) domain.

This runs the transforms behind the theory in the report:
  edges are high spatial frequencies, so a high-pass filter reveals them,
  the Laplacian of Gaussian (LoG) is a band-pass filter whose zero crossings are
    edges,
  regions are low spatial frequencies, so a low-pass filter flattens a region
    and a threshold then separates it (the same idea used in Q1 and Q2).

All filtering multiplies the shifted spectrum F(u,v) by a transfer function
H(u,v) and inverse-transforms (the convolution theorem).

Run:
    python -m src.fourier --image data/person_rgb.png --out outputs
"""

import argparse
import os
import cv2
import numpy as np


def _freq_grid(shape):
    """Distance of each frequency from the centre, after fftshift."""
    h, w = shape
    v = np.arange(h) - h // 2
    u = np.arange(w) - w // 2
    U, V = np.meshgrid(u, v)
    return np.sqrt(U ** 2 + V ** 2)


def spectrum_image(gray):
    """Log-magnitude spectrum for display."""
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.log1p(np.abs(F))
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def gaussian_highpass(gray, d0=20):
    """
    Gaussian high-pass filter, H(u,v) = 1 - exp(-D^2 / (2 d0^2)).
    Keeps high frequencies (edges) and drops the smooth low-frequency content.
    """
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    D = _freq_grid(gray.shape)
    H = 1.0 - np.exp(-(D ** 2) / (2.0 * d0 ** 2))
    out = np.fft.ifft2(np.fft.ifftshift(F * H)).real
    return cv2.normalize(np.abs(out), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def log_edges(gray, sigma=2.0):
    """
    Laplacian of Gaussian in the frequency domain.
    H(u,v) = -4 pi^2 D^2 * exp(-2 pi^2 sigma^2 D^2), a band-pass filter.
    Edges are the zero crossings of the result.
    """
    h, w = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    # normalised frequencies so the constants match the continuous formula
    v = (np.arange(h) - h // 2) / h
    u = (np.arange(w) - w // 2) / w
    U, V = np.meshgrid(u, v)
    D2 = U ** 2 + V ** 2
    H = -4.0 * np.pi ** 2 * D2 * np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * D2)
    lap = np.fft.ifft2(np.fft.ifftshift(F * H)).real

    # mark sign changes between neighbouring pixels as edges
    zc = np.zeros_like(lap, dtype=np.uint8)
    s = np.sign(lap)
    zc[:-1, :][(s[:-1, :] * s[1:, :]) < 0] = 255
    zc[:, :-1][(s[:, :-1] * s[:, 1:]) < 0] = 255
    return zc


def lowpass_then_threshold(gray, d0=25):
    """
    Region segmentation in the frequency domain. A Gaussian low-pass filter,
    H(u,v) = exp(-D^2 / (2 d0^2)), flattens each region, then Otsu separates it.
    """
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    D = _freq_grid(gray.shape)
    H = np.exp(-(D ** 2) / (2.0 * d0 ** 2))
    smooth = np.fft.ifft2(np.fft.ifftshift(F * H)).real
    smooth = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, seg = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return smooth, seg


def run(image_path, out_dir="outputs"):
    """Build the Q3 demonstration figure for one image."""
    os.makedirs(out_dir, exist_ok=True)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    spec = spectrum_image(gray)
    hp = gaussian_highpass(gray, d0=20)
    log = log_edges(gray, sigma=2.0)
    smooth, seg = lowpass_then_threshold(gray, d0=25)

    def lab(img, text):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return img

    row1 = np.hstack([lab(gray.copy(), "input"),
                      lab(spec, "|F(u,v)| spectrum"),
                      lab(hp, "Gaussian high-pass (edges)")])
    row2 = np.hstack([lab(log, "LoG zero-crossings"),
                      lab(smooth, "low-pass (regions)"),
                      lab(seg, "low-pass + Otsu")])
    fig = np.vstack([row1, row2])
    cv2.imwrite(os.path.join(out_dir, "fourier_demo.png"), fig)
    print(f"[fourier] wrote {out_dir}/fourier_demo.png")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fourier-domain edges and regions (Q3).")
    ap.add_argument("--image", default="data/person_rgb.png")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    run(args.image, args.out)
