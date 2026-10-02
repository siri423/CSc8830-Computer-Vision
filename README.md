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
| 5 | Finding a person in a video with Gemini | [Week5_Challenge](Week5_Challenge) |

Each project folder has a README and a Streamlit app. Reports and presentations
are included where the assignment calls for them.

Week 5 includes a three-slide presentation, a project report, and a Streamlit
prototype that uses the Gemini API. See the [Week 5 folder](Week5_Challenge).

## Live web apps

| Assignment | App |
|---|---|
| Module 4 | [Human Segmentation](https://csc8830-module4-human-seg.streamlit.app) |
| Week 5 | [ClipCheck Video Understanding](https://csc8830-week5-clipcheck.streamlit.app) |

The Week 5 app reads `GEMINI_API_KEY` from its Streamlit Secrets settings.

## Running any module locally

```bash
pip install -r requirements.txt
cd Module4_HumanSegmentation      # or Module2..., Module3...
streamlit run app.py
```

Needs Python 3.10 or newer.
