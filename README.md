# CSc 8830 Computer Vision

Coursework for CSc 8830 (Computer Vision) at Georgia State University. Each module
is a self-contained project with its own code, report and web app. Everything is
in this one repository.

Author: Sirichandana Bikkasani

## Modules

| Module | Topic | Folder |
|---|---|---|
| 2 | Camera calibration and object measurement | [Module2_CameraCalibration](Module2_CameraCalibration) |
| 3 | Image blurring: spatial filtering vs the Fourier domain | [Module3_ImageFiltering](Module3_ImageFiltering) |
| 4 | Finding the boundary of a human in RGB and thermal images | [Module4_HumanSegmentation](Module4_HumanSegmentation) |

Each folder has its own README with the details, the report PDF, and a Streamlit
web app (`app.py`).

## Live web app (Module 4)

The Module 4 app is deployed here: https://csc8830-module4-human-seg.streamlit.app

## Running any module locally

```bash
pip install -r requirements.txt
cd Module4_HumanSegmentation      # or Module2..., Module3...
streamlit run app.py
```

Needs Python 3.10 or newer.
