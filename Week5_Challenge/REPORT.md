# Week 5 Challenge 1: Video person search

## Summary

This project explores whether a video language model can help a reviewer find when a person appears in a clip. The idea follows the broad direction of Keynote 6, which discusses detailed video understanding and grounding video content from text queries. The prototype uses Gemini 3.8 Flash through its video API and presents the interaction in a Streamlit web app.

## Problem statement

Finding a person or event in a longer clip often requires repeated scrubbing. The task becomes harder when a person is small, partly occluded, or difficult to distinguish from the background.

**Can a Video-LLM identify an approximate time range when a person is visible and describe the visual evidence so a reviewer can check the result?**

## Keynote connection

Keynote 6 is Fahad Khan’s “Towards Detailed Video Understanding in Generative AI Era.” The workshop description covers detailed visual semantics and spatiotemporal grounding conditioned on text. This prototype narrows that direction to a time retrieval question about a person in a short clip. Gemini is the model selected for implementation; it is not presented as the model used in the keynote. A personal takeaway from watching the talk in class should be added before the presentation.

## Prototype

The app accepts a video and a natural-language question. It uploads the clip to Gemini, waits for video processing, asks the model to find the person-visible interval, and displays a text answer with approximate timestamps. It also lets the user download the answer. The current interface accepts MP4, MOV, AVI, WebM, MPEG, and 3GP files up to 20 MB.

The app is deployed at [csc8830-week5-clipcheck.streamlit.app](https://csc8830-week5-clipcheck.streamlit.app). Its source and dependencies are in [`Week5_Challenge`](https://github.com/siri423/CSc8830-Computer-Vision/tree/main/Week5_Challenge). The deployed app returned a successful response for a seven-second warehouse clip.

## Feasibility check

For the demonstration, Gemini returned a person-visible interval and description for the warehouse clip. A manual interval still needs to be marked by reviewing the full clip before calculating timestamp error. The available screenshots show the person at 00:00 and 00:01; these frames alone do not establish the full reference interval.

For the start and end points, calculate absolute timestamp error:

```text
start error = |model start − reference start|
end error   = |model end − reference end|
```

Also record whether Gemini identified the intended person and described the visible action correctly. One clip is a demonstration, not a general accuracy estimate. Add the observed timestamps and errors here after the live run:

| Measure | Observed result |
|---|---|
| Clip length | 7 seconds |
| Gemini interval | 00:00–00:07 |
| Gemini description | A worker in a warehouse checks the shelves and interacts with a tablet. |
| Model call | Completed successfully; result shown in the app screenshot |
| Manually reviewed interval | Not recorded; review the full clip before presenting a reference interval |
| Start and end error | Not measured |
| Person and action identified correctly | The screenshots at 00:00 and 00:01 show the person and warehouse setting; full-clip action verification remains a manual check |

## Data handling

The selected clip is sent to Google for analysis, so the app asks the user to confirm permission before sending it. The code requests that the Gemini interaction not be stored and deletes the uploaded Files API item after the request. Do not upload private footage or a clip without permission to share it.

## References

- [CVPR 2025 Video-LLM Workshop program](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/program.html)
- [Keynote 6 speaker description](https://www.crcv.ucf.edu/cvpr2025-vidllms-workshop/speakers.html)
- [Gemini video understanding documentation](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini 3.8 Flash model information](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Gemini API pricing and data use](https://ai.google.dev/gemini-api/docs/pricing)
