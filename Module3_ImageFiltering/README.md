# CSc 8830 Computer Vision, Module 3
## Image blurring by filtering: spatial domain vs Fourier domain

A small web app that blurs an image using a filtering approach (convolution with
a blur kernel, written by hand) and then shows the Convolution Theorem in action:
blurring in the spatial domain gives the same result as multiplying the Fourier
transforms in the frequency domain.

**Live web app:** https://module3-imagefiltering-cirvxeybpqowzzg5bfwfyt.streamlit.app/

Author: Sirichandana Bikkasani, Georgia State University

## Where each assignment requirement is covered

| Requirement | Where |
|---|---|
| Blur an image (filtering approach) | `src/spatial.py` and the web app tab "1. Blur an image" |
| Spatial filtering equals the Fourier equivalent | `src/frequency.py`, `src/compare.py` and tab "2. Spatial = Frequency" |
| Convolution in space equals multiplication in frequency | tab "2. Spatial = Frequency" and tab "3. Theory" |
| Theory and derivation | `Module3_Report.pdf` (Sections 3 and 4) and tab "3. Theory" |
| Working demo as a web application | `app.py` (Streamlit) |

## Folder layout
```
Module3_ImageFiltering/
  app.py               the Streamlit web app, everything is reachable here
  requirements.txt
  README.md
  Module3_Report.pdf   the report: theory, worked example, and the results
  Module3_Report.tex   LaTeX source for the report
  src/
    kernels.py         builds the box and Gaussian kernels
    spatial.py         the hand written 2D convolution (the blur)
    frequency.py       the FFT based version (multiply in frequency)
    compare.py         runs the spatial vs frequency experiment and makes figures
    utils.py           load and save images
  data/
    sample.jpg         the built in test image (you can also upload your own)
  outputs/             the result figures used in the report
```

## Setup
```
pip install -r requirements.txt
```

## Run the web app (this is what the screen recording shows)
```
streamlit run app.py
```
Then open the address it prints (usually http://localhost:8501) and go through
the tabs:

1. Overview: what the app does and the link to the GitHub repo.
2. Blur an image: upload a picture or use the built in sample, pick a kernel
   (box or Gaussian) and a size, and see it blurred. There is a checkbox to
   process in color if you want.
3. Spatial = Frequency: the same image is blurred both ways and the app reports
   the biggest difference (around 1e-15, so they are the same), with a side by
   side view and the frequency spectra.
4. Theory: the Convolution Theorem written out.

## Run from the command line (optional, reproduces the figures)
```
python -m src.compare --kind gaussian --size 15
python -m src.compare --kind box --size 9
```
Example output:
```
max abs diff   : 2.665e-15      spatial and frequency results are the same
MSE            : 2.226e-31
PSNR (dB)      : 306.5
```

## How it works in short
A blur kernel (box or Gaussian, built in `kernels.py`) is a small grid of weights
that add up to 1. Blurring convolves the image with that kernel. `spatial.py` does
this directly with a convolution I wrote by hand, no library blur function.
`frequency.py` does the same job in the Fourier domain: it takes the FFT of the
image and the kernel, multiplies them, and transforms back, because convolution in
space is the same as multiplication in frequency. `compare.py` runs both on the
same input and measures the gap, which sits around 1e-15, so the theorem holds.

## Requirements
Python 3.9 or newer, plus streamlit, numpy, pillow and matplotlib (see
`requirements.txt`).
