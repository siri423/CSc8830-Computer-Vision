# CSc 8830 Computer Vision, Module 4
## Finding the boundary of a human in RGB and thermal images

This project outlines a person in a normal colour photo and in a thermal
(infrared) image using classical computer vision only. Everything in the
segmentation code is plain OpenCV: thresholding, morphology, connected
components, contours and GrabCut. There is no deep learning or machine learning
in the segmentation. The results are then placed next to the Segment Anything
model, which the assignment asks us to compare against.

Author: Sirichandana Bikkasani, Georgia State University

**Live app:** https://csc8830-module4-human-seg.streamlit.app
**Repository:** https://github.com/siri423/CSc8830-Computer-Vision

## What each part of the assignment maps to

| Requirement | Where it is |
|---|---|
| Q1, person boundary in an RGB image (classical) | `src/rgb_segment.py`, web app tab "RGB image" |
| Q2, person boundary in a thermal image (classical) | `src/thermal_segment.py`, web app tab "Thermal image" |
| Compare with SAM | `src/compare.py`, `outputs/*_compare.png`, `outputs/metrics.json`, web app tab "Compare with SAM" |
| Q3, theory of edges and regions in the Fourier domain | `Module4_Report.pdf` (Section 5), `src/fourier.py`, web app tab "Theory" |
| Working demo as a web app | `app.py` (Streamlit) |

## Folder layout

```
Module4_HumanSegmentation/
  app.py                 the Streamlit web app, everything is reachable here
  requirements.txt
  README.md
  Module4_Report.pdf     the report: method, results, comparison and theory
  Module4_Report.tex     LaTeX source for the report
  .streamlit/config.toml theme for the web app
  src/
    rgb_segment.py       Q1, RGB boundary with GrabCut
    thermal_segment.py   Q2, thermal boundary with thresholding and morphology
    compare.py           IoU, Dice and boundary-F against the SAM masks, plus figures
    fourier.py           Q3, edges and regions in the frequency domain
    utils.py             image reading and drawing helpers
  data/
    person_rgb.png       RGB sample (NASA astronaut, public domain)
    person_thermal.png   thermal sample (OSU thermal pedestrian frame)
  sam2/
    rgb_sam.png          SAM reference mask for the RGB image
    thermal_sam.png      SAM reference mask for the thermal image
  outputs/               result masks, overlays, comparison figures, metrics
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

Needs Python 3.10 or newer, plus OpenCV, NumPy, Matplotlib and Streamlit.

## Running it

Web app, which is the easiest way to see everything:

```bash
streamlit run app.py
```

Open the link it prints. You can use the built-in samples or upload your own
image on the RGB, Thermal and Theory tabs.

From the command line:

```bash
python -m src.rgb_segment      --image data/person_rgb.png     --out outputs
python -m src.thermal_segment  --image data/person_thermal.png --out outputs
python -m src.compare
python -m src.fourier          --image data/person_rgb.png     --out outputs
```

## Results, our classical mask vs SAM

| Image | IoU | Dice | Boundary-F @2px |
|---|---|---|---|
| RGB (astronaut) | 0.68 | 0.81 | 0.52 |
| Thermal (pedestrian) | 0.82 | 0.90 | 0.95 |

On the thermal image the classical pipeline almost matches SAM, because a warm
body is easy to pull out from a cool background. On the colour photo the
classical result is close on the overall shape but its fine boundary is less
accurate than SAM, which is the kind of gap learned models are built to close.

## Where the sample images come from

- RGB: the `astronaut` image, a portrait of NASA astronaut Eileen Collins. It is
  public domain and ships with scikit-image.
- Thermal: a frame from the OSU Thermal Pedestrian Database (OTCBVS Benchmark), a
  standard academic thermal dataset, pulled from a public GitHub mirror. Used
  here for coursework.

## Note on the SAM comparison

The assignment names SAM2. SAM2's own checkpoints and hosted demo were not
reachable from the machine used to build this, so the reference here is the SAM3
demo, which is the current model in the same Segment Anything family, run with
the text prompt "person". The comparison does not depend on the exact version. To
redo it with the official SAM2 demo, segment the same two images
(`data/person_rgb.png` and `data/person_thermal.png`), save the masks as
`sam2/rgb_sam.png` and `sam2/thermal_sam.png`, and run `python -m src.compare`.
