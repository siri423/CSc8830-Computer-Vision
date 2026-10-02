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

Each folder has its own README with the details, the report PDF, and a Streamlit
web app (`app.py`).

Week 5 includes a three-slide presentation and a Streamlit prototype that uses
the Gemini API. Add the API key through Streamlit Secrets before running it;
see [the Week 5 setup guide](Week5_Challenge/SUBMISSION_GUIDE.md).

## Live web app (Module 4)

The Module 4 app is deployed here: https://csc8830-module4-human-seg.streamlit.app

The Week 5 app needs its own Streamlit deployment and a `GEMINI_API_KEY` secret.

## Running any module locally

```bash
pip install -r requirements.txt
cd Module4_HumanSegmentation      # or Module2..., Module3...
streamlit run app.py
```

Needs Python 3.10 or newer.
