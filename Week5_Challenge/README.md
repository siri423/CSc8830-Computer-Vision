# Week 5 Challenge 1: Finding a person in a video

This project explores a practical question inspired by Keynote 6: can a video model help someone find when a person appears in a clip, without watching the whole clip repeatedly?

Keynote 6 was Fahad Khan’s “Towards Detailed Video Understanding in Generative AI Era.” The workshop description discusses detailed visual semantics and spatiotemporal grounding from text queries. That motivated this small experiment. The prototype uses Google Gemini 3.8 Flash because it accepts video input through an API; it is not presented as a model used in the keynote. I still need to watch the talk in class and add my own takeaway before submitting.

## What the prototype does

The Streamlit app sends one short video and a text question to Gemini. It asks for an approximate time range when the main person is clearly visible, a brief description, and visual evidence. The answer is text with estimated timestamps. It does not create a segmentation mask, track a person across frames, or establish ground truth.

The app is deployed at [csc8830-week5-clipcheck.streamlit.app](https://csc8830-week5-clipcheck.streamlit.app). A successful response was returned for a seven-second warehouse clip: Gemini reported 00:00–00:07 and described a worker checking shelves and using a tablet. The manual reference interval and timestamp error have not been measured. Uploads are limited to 20 MB. Gemini must finish processing a video before answering, and processing time depends on the clip and service availability.

The folder also contains the [three-slide presentation](output/Week5_Challenge1_Presentation_v2.pptx), [three-minute recording script](recording_script.md), [project report](REPORT.md), and [submission guide](SUBMISSION_GUIDE.md).

## Run locally

Use Python 3.10 or newer. From this folder, install the packages:

```bash
python3 -m pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` in this folder with your own key:

```toml
GEMINI_API_KEY = "your-key-here"
```

Do not commit this file or share the key. It is excluded by `.gitignore`. Then start the app:

```bash
python3 -m streamlit run app.py
```

The app supports MP4, MOV, AVI, WebM, MPEG and 3GP files up to 20 MB. Short MP4 clips are the safest choice. Check the answer by replaying the clip, since timestamps can be approximate and brief actions can be missed.

## Video handling

The app sends the selected clip to Google for analysis. It requests that the interaction not be stored and deletes the uploaded Files API resource after the request. Google documents automatic expiration of uploaded files after 48 hours as a fallback. Free-tier data may be used to improve Google products, so do not upload private or sensitive footage. Upload only a clip you have permission to share.

## Selected keynote and references

- Keynote 6: Fahad Khan, “Towards Detailed Video Understanding in Generative AI Era.” [Workshop program](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/program.html) and [speaker description](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/speakers.html).
- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini 3.8 Flash model information](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Gemini API pricing and data use](https://ai.google.dev/gemini-api/docs/pricing)
