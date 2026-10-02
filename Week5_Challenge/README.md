# Week 5 Challenge: Video person search

This project tests whether a video language model can help a reviewer find when a person appears in a short clip. It was inspired by Keynote 6, Fahad Khan’s “Towards Detailed Video Understanding in Generative AI Era,” which discusses detailed video understanding and grounding visual content from text. The prototype explores one focused question: can a natural-language query return an approximate person-visible interval and a description that a reviewer can check?

Gemini 3.8 Flash was selected for the prototype because its API accepts video input. It is the implementation choice for this project; it is not presented as the model used in the keynote.

## Prototype and result

The Streamlit app accepts a short video and a text question, sends them to Gemini, and displays an approximate time range and a brief description. A successful run on a seven-second warehouse clip returned 00:00–00:07 and described a worker checking shelves and using a tablet.

- **Live app:** [ClipCheck](https://csc8830-week5-clipcheck.streamlit.app)
- **Presentation:** [Three-slide deck](output/Week5_Challenge1_Presentation_v5.pptx)
- **Report:** [Problem, keynote connection, and feasibility results](REPORT.md)

The example demonstrates that the end-to-end request works. Timestamp accuracy has not been measured against a manually annotated reference interval, and this single clip is not a general performance evaluation.

## Run locally

Use Python 3.10 or newer. From this folder, install the packages and launch the app:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Set `GEMINI_API_KEY` in Streamlit Secrets or in your local environment before running an analysis. Do not commit API keys. The app accepts MP4, MOV, AVI, WebM, MPEG, and 3GP files up to 20 MB. A short MP4 is recommended for a quick demonstration.

## Data handling

The selected clip is sent to Google for analysis. The app asks for permission before sending it and deletes the uploaded Files API item after the request. Upload only video that you have permission to share, and avoid private or sensitive footage.

## References

- [CVPR 2025 Video-LLM Workshop program](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/program.html)
- [Keynote 6 speaker description](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/speakers.html)
- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini 3.8 Flash model information](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
