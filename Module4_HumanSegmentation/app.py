"""
CSc 8830 Module 4 web application.

One Streamlit app that reaches every part of the assignment:
  Overview        what the app does and the results at a glance
  RGB image       Q1, classical GrabCut boundary, no ML
  Thermal image   Q2, classical threshold and morphology boundary, no ML
  Compare w/ SAM  our classical result against the Segment Anything reference
  Theory          Q3, edges and regions in the frequency domain

Run it:
    streamlit run app.py
"""

import json
import os
import cv2
import numpy as np
import streamlit as st

from src.rgb_segment import segment_rgb
from src.thermal_segment import segment_thermal
from src.fourier import (spectrum_image, gaussian_highpass, log_edges,
                         lowpass_then_threshold)
from src.utils import draw_boundary, cutout, imread_color, OURS_BGR, SAM_BGR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
SAM = os.path.join(HERE, "sam2")

ACCENT = "#1E90FF"       # blue, our result
SAM_HEX = "#2ECC71"      # green, SAM reference

st.set_page_config(page_title="Module 4 - Human Boundary Segmentation",
                   layout="wide")

# A little styling so the page looks tidy and consistent.
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1150px; }}
  h1, h2, h3 {{ color: #14202c; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
      padding: 10px 16px; border-radius: 8px 8px 0 0; font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{ background: {ACCENT}1a; color: {ACCENT}; }}
  .legend {{ display:inline-block; padding:2px 10px; border-radius:12px;
             font-size:0.85rem; font-weight:600; margin-right:8px; }}
  .card {{ background:#f4f7fb; border:1px solid #e3e9f0; border-radius:10px;
           padding:14px 18px; margin-bottom:10px; }}
  .stButton>button {{ border-radius:8px; font-weight:600; }}
</style>
""", unsafe_allow_html=True)


# small helpers
def to_rgb(img):
    """BGR or grayscale to RGB for display."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png_bytes(img):
    """Encode an image to PNG bytes for a download button."""
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def image_input(key, sample_path, gray=False, upload_types=("png", "jpg", "jpeg", "bmp")):
    """
    Reusable input block: pick the built-in sample or upload your own, with a
    preview. Returns the chosen image (BGR, or grayscale if gray=True).
    """
    mode = st.radio("Image source", ["Built-in sample", "Upload your own"],
                    horizontal=True, key=f"mode_{key}")
    img = None
    if mode == "Upload your own":
        up = st.file_uploader("Choose an image file", type=list(upload_types),
                              key=f"up_{key}",
                              help="PNG, JPG or BMP. Kept only for this session.")
        if up is not None:
            buf = np.frombuffer(up.read(), np.uint8)
            flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
            img = cv2.imdecode(buf, flag)
        else:
            st.info("Upload an image, or switch back to the built-in sample.")
    else:
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imread(sample_path, flag)
    return img


def result_row(bgr, mask, key):
    """Show boundary, cut-out, and mask side by side with download buttons."""
    boundary = draw_boundary(bgr, mask, OURS_BGR, 2)
    cut = cutout(bgr, mask)
    c1, c2, c3 = st.columns(3)
    c1.image(to_rgb(boundary), caption="Detected boundary", use_container_width=True)
    c2.image(to_rgb(cut), caption="Background removed", use_container_width=True)
    c3.image(mask, caption="Binary mask", use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download boundary", png_bytes(boundary),
                       file_name=f"{key}_boundary.png", key=f"db_{key}")
    d2.download_button("Download cut-out", png_bytes(cut),
                       file_name=f"{key}_cutout.png", key=f"dc_{key}")
    d3.download_button("Download mask", png_bytes(mask),
                       file_name=f"{key}_mask.png", key=f"dm_{key}")
    pct = 100 * cv2.countNonZero(mask) / mask.size
    st.success(f"Done. The person covers {pct:.1f}% of the image.")


# header
st.title("Finding the Exact Boundary of a Human")
st.caption("CSc 8830 Computer Vision, Module 4  ·  Sirichandana Bikkasani, "
           "Georgia State University")
st.markdown(
    f'<span class="legend" style="background:{ACCENT}22;color:{ACCENT};">'
    f'Blue = our classical result</span>'
    f'<span class="legend" style="background:{SAM_HEX}22;color:{SAM_HEX};">'
    f'Green = SAM reference</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("This app finds the outline of a person in a colour photo and in a "
             "thermal image using classical computer vision only. No deep "
             "learning is used for the segmentation. The Segment Anything model "
             "is shown alongside as a reference.")
    st.write("Use the built-in samples, or upload your own image on the RGB, "
             "Thermal and Theory tabs.")
    st.divider()
    st.caption("Built with OpenCV, NumPy and Streamlit.")

tabs = st.tabs(["Overview", "RGB image (Q1)", "Thermal image (Q2)",
                "Compare with SAM", "Theory (Q3)"])

# Overview
with tabs[0]:
    st.subheader("What this project does")
    st.write("A person is outlined in two very different kinds of image. In a "
             "colour photo the separation is based on colour (GrabCut). In a "
             "thermal image it is based on how warm the body is compared with "
             "the ground. Both methods use classical OpenCV only.")

    st.subheader("How our results line up with SAM")
    mpath = os.path.join(OUT, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            metrics = json.load(f)
        st.table({m["image"].upper(): {"IoU": m["IoU"], "Dice": m["Dice"],
                                       "Boundary-F @2px": m["BoundaryF@2px"]}
                  for m in metrics})
    st.write("On the thermal image the classical method almost matches SAM, "
             "because a warm body stands out clearly from a cool background. On "
             "the colour photo it captures the person but the fine edge is less "
             "accurate than SAM, which is the kind of case learned models handle "
             "better.")

    st.subheader("Where each requirement lives")
    st.markdown("""
- **Q1, RGB boundary** — the *RGB image* tab and `src/rgb_segment.py`
- **Q2, thermal boundary** — the *Thermal image* tab and `src/thermal_segment.py`
- **Compare with SAM** — the *Compare with SAM* tab and `src/compare.py`
- **Q3, Fourier theory** — the *Theory* tab, `src/fourier.py`, and the report PDF
""")

# Q1 RGB
with tabs[1]:
    st.subheader("Q1 - Human boundary in a colour (RGB) image")
    st.write("Method: automatic GrabCut. GrabCut fits colour mixtures for the "
             "foreground and background and cuts the cheapest boundary between "
             "them. It learns those colours from this one image, so no trained "
             "model is involved. We seed it automatically, clean the mask with "
             "morphology, keep the largest region and fill holes.")
    bgr = image_input("rgb", os.path.join(DATA, "person_rgb.png"), gray=False)
    if bgr is not None:
        st.image(to_rgb(bgr), caption="Input", width=360)
        if st.button("Find the boundary", type="primary", key="run_rgb"):
            with st.spinner("Running GrabCut..."):
                mask = segment_rgb(bgr)
            result_row(bgr, mask, "rgb")

# Q2 Thermal
with tabs[2]:
    st.subheader("Q2 - Human boundary in a thermal image")
    st.write("Method: a two-stage approach. First locate the person by keeping "
             "the hottest regions and scoring each hot blob on how human it "
             "looks (upright shape, standing on the ground, surrounded by cool "
             "pixels). Then refine the outline with a local threshold around the "
             "person so the dimmer legs are included. Classical OpenCV only.")
    gray = image_input("th", os.path.join(DATA, "person_thermal.png"), gray=True)
    if gray is not None:
        st.image(to_rgb(gray), caption="Input", width=360)
        if st.button("Find the boundary", type="primary", key="run_th"):
            with st.spinner("Segmenting..."):
                mask = segment_thermal(gray)
            result_row(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mask, "thermal")

# Q3 Compare
with tabs[3]:
    st.subheader("Comparison with SAM")
    st.write("Our classical outline is drawn in blue and the Segment Anything "
             "outline in green on the same image. SAM is only the reference the "
             "assignment asks us to compare against; our own code never uses it.")
    mpath = os.path.join(OUT, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            metrics = json.load(f)
        st.table({m["image"].upper(): {"IoU": m["IoU"], "Dice": m["Dice"],
                                       "Boundary-F @2px": m["BoundaryF@2px"]}
                  for m in metrics})
    for name, title in [("rgb", "RGB"), ("thermal", "Thermal")]:
        p = os.path.join(OUT, f"{name}_compare.png")
        if os.path.exists(p):
            st.image(p, caption=f"{title}: blue is our result, green is SAM",
                     use_container_width=True)

# Q3 Theory
with tabs[4]:
    st.subheader("Q3 - Edges and regions in the Fourier domain")
    st.write("Smooth areas of an image are low spatial frequencies. Sharp "
             "changes, which is what edges are, are high spatial frequencies. "
             "That single fact is enough to do both edge detection and region "
             "segmentation by filtering the spectrum.")

    st.markdown("##### 1. The Fourier transform")
    st.write("For an image f(x, y):")
    st.latex(r"F(u,v)=\iint f(x,y)\,e^{-j2\pi(ux+vy)}\,dx\,dy")
    st.write("Low frequencies sit near the centre of F, high frequencies away "
             "from it.")

    st.markdown("##### 2. Edges are a high-pass filter")
    st.write("Taking a derivative multiplies the spectrum by a factor that grows "
             "with frequency, so any edge operator keeps high frequencies:")
    st.latex(r"\mathcal{F}\!\left\{\tfrac{\partial f}{\partial x}\right\}=j2\pi u\,F(u,v)"
             r"\qquad \mathcal{F}\{\nabla^2 f\}=-4\pi^2(u^2+v^2)\,F(u,v)")
    st.write("A Gaussian high-pass filter keeps those edges, with "
             "D the distance from the centre of the spectrum:")
    st.latex(r"H_{\text{hp}}(u,v)=1-e^{-D^2/(2D_0^2)},\qquad D=\sqrt{u^2+v^2}")

    st.markdown("##### 3. Laplacian of Gaussian, a band-pass edge detector")
    st.write("Smoothing first with a Gaussian and then applying the Laplacian "
             "gives a band-pass filter. Its zero crossings are the edges "
             "(Marr and Hildreth):")
    st.latex(r"H_{\text{LoG}}(u,v)=-4\pi^2 D^2\,e^{-2\pi^2\sigma^2 D^2}")

    st.markdown("##### 4. Regions are a low-pass filter plus a threshold")
    st.write("A low-pass filter flattens each region so a simple threshold can "
             "separate it, which is the same smoothing then threshold idea used "
             "on the thermal image:")
    st.latex(r"H_{\text{lp}}(u,v)=e^{-D^2/(2D_0^2)},\qquad "
             r"f_{\text{lp}}=\mathcal{F}^{-1}\{F\,H_{\text{lp}}\}")
    st.write("High-pass and low-pass are complementary halves of the same "
             "spectrum, since H_hp = 1 - H_lp. One pulls out the boundary between "
             "regions, the other pulls out the smooth interior.")

    st.markdown("##### 5. Texture regions with a Gabor filter bank")
    st.write("When regions differ by texture rather than brightness, a Gabor "
             "filter (a Gaussian centred on a chosen frequency) selects one "
             "orientation and scale:")
    st.latex(r"H(u,v)=e^{-2\pi^2\sigma^2\left[(u-u_0)^2+(v-v_0)^2\right]}")
    st.write("Filtering with a bank of these gives every pixel a frequency "
             "signature, and pixels with similar signatures are grouped into "
             "texture regions. Full derivations are in the report PDF.")

    st.divider()
    st.markdown("##### Live demonstration")
    st.write("Watch the same image become edges (high-pass) or regions "
             "(low-pass). Adjust the cut-off and re-run.")
    gray = image_input("ft", os.path.join(DATA, "person_rgb.png"), gray=True)
    c1, c2 = st.columns(2)
    d0 = c1.slider("Filter cut-off D0 (pixels)", 5, 60, 20)
    sigma = c2.slider("LoG sigma", 0.5, 5.0, 2.0)
    if gray is not None and st.button("Run the Fourier demo", type="primary",
                                      key="run_ft"):
        smooth, seg = lowpass_then_threshold(gray, d0=d0 + 5)
        r1 = st.columns(3)
        r1[0].image(to_rgb(gray), caption="Input", use_container_width=True)
        r1[1].image(spectrum_image(gray), caption="Spectrum |F(u,v)|",
                    use_container_width=True, clamp=True)
        r1[2].image(gaussian_highpass(gray, d0=d0), caption="High-pass, edges",
                    use_container_width=True, clamp=True)
        r2 = st.columns(3)
        r2[0].image(log_edges(gray, sigma=sigma), caption="LoG zero-crossings",
                    use_container_width=True, clamp=True)
        r2[1].image(smooth, caption="Low-pass, regions",
                    use_container_width=True, clamp=True)
        r2[2].image(seg, caption="Low-pass then Otsu",
                    use_container_width=True, clamp=True)
    else:
        demo = os.path.join(OUT, "fourier_demo.png")
        if os.path.exists(demo):
            st.image(demo, caption="Precomputed demonstration",
                     use_container_width=True)
