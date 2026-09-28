"""
app.py   CSc 8830 Computer Vision, Module 3

This is the web app. Everything for the assignment is reachable from the tabs
on the page:

    Overview             what the app is and the link to the GitHub repo
    1. Blur an image     blur a picture using the filtering approach
    2. Spatial = Freq    show that spatial convolution == frequency multiplication
    3. Theory            the Convolution Theorem written out

Start it with:
    streamlit run app.py
then open the address it prints (usually http://localhost:8501).
"""

import io
import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# our own code. the actual blurring lives in these files, no library blur is used
from src.kernels import make_kernel
from src.spatial import blur_spatial
from src.frequency import blur_frequency, magnitude_spectrum
from src.compare import compare_methods
from src.utils import load_image, to_uint8, apply_per_channel

# change this to your repo link once the repo is created on GitHub
GITHUB_URL = "https://github.com/siri423/Module3-ImageFiltering"
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample.jpg")

st.set_page_config(page_title="CSc 8830 Module 3 Image Filtering", layout="wide")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_upload(file_bytes, grayscale):
    """Turn the bytes from an uploaded file into a float image in 0..1."""
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes))
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def get_input_image(grayscale, key):
    """Let the user upload a picture, or fall back to the built in sample."""
    up = st.file_uploader("Upload your own image (optional)",
                          type=["jpg", "jpeg", "png", "bmp"], key=key)
    if up is not None:
        return _read_upload(up.getvalue(), grayscale)
    return load_image(SAMPLE_PATH, grayscale=grayscale)


def kernel_controls(prefix):
    """The kernel type, size and sigma controls. Returns (kind, size, sigma)."""
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Kernel type", ["gaussian", "box"], key=prefix + "kind")
    size = c2.slider("Kernel size (odd)", 3, 41, 15, step=2, key=prefix + "size")
    sigma = None
    if kind == "gaussian":
        sigma = c3.slider("Gaussian sigma", 0.5, 15.0, float(size) / 6.0,
                          step=0.5, key=prefix + "sigma")
    else:
        c3.markdown("*(box kernel has no sigma)*")
    return kind, size, sigma


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.title("CSc 8830 Computer Vision, Module 3")
st.subheader("Image blurring by filtering: spatial domain vs Fourier domain")

tabs = st.tabs(["Overview",
                "1. Blur an image",
                "2. Spatial = Frequency (proof)",
                "3. Theory"])

# ==========================================================================
# Overview
# ==========================================================================
with tabs[0]:
    st.markdown("""
### What this app does
This app blurs an image using a filtering approach, and then shows a basic
result from signal processing, the **Convolution Theorem**:

> Convolving an image with a filter in the spatial domain gives the exact same
> result as multiplying their Fourier transforms in the frequency domain.

**Where each part of the assignment lives:**

| Requirement | Tab |
|---|---|
| Blur an image (filtering approach) | **1. Blur an image** |
| Spatial filtering equals the Fourier equivalent | **2. Spatial = Frequency** |
| Convolution in space equals multiplication in frequency | **2. Spatial = Frequency** |
| Theory | **3. Theory** (full write up is in the report PDF) |
""")
    st.markdown("**GitHub repository:** [%s](%s)" % (GITHUB_URL, GITHUB_URL))
    st.markdown("""
The blur uses a kernel we build ourselves (box or Gaussian) and a 2D convolution
we wrote by hand in `src/spatial.py`. The Fourier version in `src/frequency.py`
takes the FFT of the image and the kernel, multiplies them, and transforms back.
Tab 2 blurs the same image both ways and reports the difference, which comes out
around 1e-15 (rounding error), so the two results are the same.
""")
    st.info("For the screen recording, go through the tabs left to right: blur an "
            "image on Tab 1, then open Tab 2 to show the two methods agree, then "
            "Tab 3 for the theory.")

# ==========================================================================
# Tab 1, blur an image
# ==========================================================================
with tabs[1]:
    st.markdown("### Blur an image using a filtering approach")
    st.caption("The blur is a 2D convolution of the image with a blur kernel, "
               "written by hand in src/spatial.py.")

    color = st.checkbox("Process in color (RGB)", value=False, key="blur_color")
    img = get_input_image(grayscale=not color, key="blur_upload")
    kind, size, sigma = kernel_controls("blur_")
    kernel = make_kernel(kind, size, sigma)

    blurred = apply_per_channel(blur_spatial, img, kernel, mode="reflect")

    c1, c2 = st.columns(2)
    c1.markdown("**Original**")
    c1.image(to_uint8(img), use_container_width=True)
    label = "**Blurred**, %s %dx%d" % (kind, size, size)
    if sigma:
        label += ", sigma=%.1f" % sigma
    c2.markdown(label)
    c2.image(to_uint8(blurred), use_container_width=True)

    # show the kernel weights as a heatmap
    with st.expander("See the blur kernel (the filter weights)"):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(kernel, cmap="viridis")
        ax.set_title("%s kernel %dx%d (weights sum to 1)" % (kind, size, size))
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        st.write("Kernel sum = %.6f, min = %.5f, max = %.5f"
                 % (kernel.sum(), kernel.min(), kernel.max()))

    # let the user download the blurred picture
    from PIL import Image as _PILImage
    _buf = io.BytesIO()
    _PILImage.fromarray(to_uint8(blurred)).save(_buf, format="PNG")
    st.download_button("Download blurred image (PNG)",
                       data=_buf.getvalue(),
                       file_name="blurred_%s_%d.png" % (kind, size),
                       mime="image/png", key="dl_blurred")

# ==========================================================================
# Tab 2, spatial vs frequency proof
# ==========================================================================
with tabs[2]:
    st.markdown("### Proof by experiment: spatial convolution vs frequency multiplication")
    st.caption("Same image, same kernel, blurred two different ways. If the "
               "Convolution Theorem is true, the two results have to be identical.")

    img2 = get_input_image(grayscale=True, key="proof_upload")
    kind2, size2, sigma2 = kernel_controls("proof_")

    res = compare_methods(img2, kind2, size2, sigma2)
    m = res["metrics"]

    # the numbers
    a, b, c = st.columns(3)
    a.metric("max abs difference", "%.2e" % m["max_abs_diff"],
             help="Biggest pixel disagreement. Around 1e-15 means the same image.")
    b.metric("MSE", "%.2e" % m["mse"])
    c.metric("PSNR (dB)", "inf" if np.isinf(m["psnr_db"]) else "%.1f" % m["psnr_db"])

    if m["max_abs_diff"] < 1e-9:
        st.success("The two images are the same down to floating point precision. "
                   "Convolution in space equals multiplication in frequency.")

    # side by side pictures
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img2, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(res["spatial"], cmap="gray"); ax[1].set_title("Spatial convolution")
    ax[2].imshow(res["frequency"], cmap="gray"); ax[2].set_title("Frequency multiplication")
    d = ax[3].imshow(res["diff"], cmap="magma")
    ax[3].set_title("abs difference (max=%.1e)" % m["max_abs_diff"])
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Timing on this image: spatial %.0f ms, frequency %.0f ms."
               % (res["time_spatial"] * 1000, res["time_frequency"] * 1000))

    # frequency picture
    with st.expander("Show the frequency domain view (why blur is a low pass filter)"):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
        ax[0].imshow(magnitude_spectrum(img2), cmap="gray")
        ax[0].set_title("Image spectrum |F| (log)")
        kresp = np.fft.fftshift(np.abs(np.fft.fft2(res["kernel"], s=img2.shape)))
        kresp = kresp / kresp.max()
        ax[1].imshow(kresp, cmap="gray")
        ax[1].set_title("Kernel response |G| (low pass)")
        ax[2].imshow(magnitude_spectrum(res["frequency"]), cmap="gray")
        ax[2].set_title("Blurred spectrum |F.G| (log)")
        for x in ax:
            x.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write("The kernel response is bright in the middle (low frequencies "
                 "pass) and dark at the edges (high frequencies removed). "
                 "Multiplying by it drops the fine detail, and that is what "
                 "blurring is.")

# ==========================================================================
# Tab 3, theory
# ==========================================================================
with tabs[3]:
    st.markdown(r"""
### The Convolution Theorem

**Spatial convolution.** Blurring an image $f$ with a kernel $g$ is a 2D
convolution:

$$(f * g)(x,y) = \sum_{s}\sum_{t} f(s,t)\, g(x-s,\, y-t).$$

**Fourier transform.** The 2D Discrete Fourier Transform of an $M\times N$ image
is

$$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\,
e^{-j 2\pi (ux/M + vy/N)}.$$

**The theorem.**

$$f * g \;\;\Longleftrightarrow\;\; F \cdot G$$

Convolution in the spatial domain equals element-wise multiplication in the
frequency domain (and the other way around). So the blur can be done either way:

$$f * g \;=\; \mathcal{F}^{-1}\{\, \mathcal{F}\{f\}\cdot \mathcal{F}\{g\}\,\}.$$

**Why a blur kernel blurs.** A box or Gaussian kernel is a low pass filter. Its
transform $G(u,v)$ is large near the origin (low frequencies) and small far away
(high frequencies). Multiplying $F\cdot G$ keeps the smooth, low frequency
content and cuts the fine, high frequency detail, which is what "blur" means.

**Linear vs circular convolution.** The plain FFT gives circular convolution,
where the kernel wraps around the image edges. To reproduce ordinary zero padded
convolution we zero pad both arrays to size $(M+k-1)\times(N+k-1)$ before
transforming, then crop the middle. That is handled in `src/frequency.py`, and it
is why Tab 2 shows agreement to about $10^{-15}$.

The full derivation, with a small worked example, is in the report PDF submitted
with this assignment.
""")
