"""
compare.py
This is the experiment that checks the Convolution Theorem.

It blurs the same image with the same kernel two ways, the hand written spatial
convolution (spatial.py) and the Fourier method (frequency.py), and then measures
how far apart the two results are. If the theorem holds they should be identical
apart from tiny floating point rounding (the biggest difference comes out around
1e-15). You can also run this file from the command line to regenerate all the
figures used in the report.
"""

from __future__ import annotations
import time
import numpy as np

from .kernels import make_kernel
from .spatial import blur_spatial
from .frequency import blur_frequency
from .utils import apply_per_channel


def error_metrics(a, b):
    """
    Compare two images and return a few numbers describing the gap between them.

    max_abs_diff  the single biggest difference at any pixel. This is the main
                  one to look at. If it is near 1e-15 the two images are the same.
    mse           the average squared difference over all pixels.
    psnr_db       peak signal to noise ratio in decibels. Very large (or infinite)
                  when the two images are basically identical.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff**2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    return {"max_abs_diff": max_abs, "mse": mse, "psnr_db": psnr}


def compare_methods(image, kind="gaussian", size=15, sigma=None):
    """
    Blur the image both ways and bundle up everything the app and report need.
    Works for a gray image (H, W) or a color image (H, W, 3).
    """
    kernel = make_kernel(kind, size, sigma)

    t0 = time.perf_counter()
    spatial = apply_per_channel(blur_spatial, image, kernel, mode="zero")
    t1 = time.perf_counter()
    frequency = apply_per_channel(blur_frequency, image, kernel)
    t2 = time.perf_counter()

    metrics = error_metrics(spatial, frequency)
    return {
        "kernel": kernel,
        "spatial": spatial,
        "frequency": frequency,
        "diff": np.abs(spatial - frequency),
        "metrics": metrics,
        "time_spatial": t1 - t0,
        "time_frequency": t2 - t1,
    }


# --------------------------------------------------------------------------
# Command line entry point. Regenerates the figures into ../outputs/.
# Run it as:  python -m src.compare --kind gaussian --size 15
# --------------------------------------------------------------------------
def _main():
    import argparse
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .utils import load_image, save_image
    from .frequency import magnitude_spectrum

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Spatial vs frequency blur experiment")
    ap.add_argument("--image", default=os.path.join(here, "data", "sample.jpg"))
    ap.add_argument("--kind", default="gaussian", choices=["gaussian", "box"])
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(here, "outputs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = load_image(args.image, grayscale=True)
    res = compare_methods(img, args.kind, args.size, args.sigma)
    m = res["metrics"]

    print("Image          :", args.image, img.shape)
    print("Kernel         :", args.kind, str(args.size) + "x" + str(args.size))
    print("max abs diff   : {:.3e}".format(m["max_abs_diff"]))
    print("MSE            : {:.3e}".format(m["mse"]))
    print("PSNR (dB)      :", m["psnr_db"])
    print("time spatial   : {:.1f} ms".format(res["time_spatial"] * 1000))
    print("time frequency : {:.1f} ms".format(res["time_frequency"] * 1000))

    # plain result images
    save_image(img, os.path.join(args.out, "01_original.png"))
    save_image(res["spatial"], os.path.join(args.out, "02_blur_spatial.png"))
    save_image(res["frequency"], os.path.join(args.out, "03_blur_frequency.png"))

    # side by side figure: original, both methods, and the difference
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(res["spatial"], cmap="gray"); ax[1].set_title("Spatial convolution")
    ax[2].imshow(res["frequency"], cmap="gray"); ax[2].set_title("Frequency multiplication")
    d = ax[3].imshow(res["diff"], cmap="magma")
    ax[3].set_title("abs difference  (max={:.1e})".format(m["max_abs_diff"]))
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle("Spatial vs frequency blur, " + args.kind + " "
                 + str(args.size) + "x" + str(args.size), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # frequency picture: image spectrum, kernel response, blurred spectrum
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].imshow(magnitude_spectrum(img), cmap="gray")
    ax[0].set_title("Image spectrum |F(u,v)| (log)")
    kresp = np.fft.fftshift(np.abs(np.fft.fft2(res["kernel"], s=img.shape)))
    kresp = kresp / kresp.max()
    ax[1].imshow(kresp, cmap="gray"); ax[1].set_title("Kernel response |G(u,v)| (low pass)")
    ax[2].imshow(magnitude_spectrum(res["frequency"]), cmap="gray")
    ax[2].set_title("Blurred spectrum |F.G| (log)")
    for a in ax:
        a.axis("off")
    fig.suptitle("Why blurring is low pass filtering: high frequencies are removed",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_spectra.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\nFigures written to:", args.out)


if __name__ == "__main__":
    _main()
