# Three-minute presentation and app demo

Show the three slides first, then switch to the deployed ClipCheck app. The quoted text is the narration. The one bracketed sentence is for you to complete after watching Keynote 6 in class; replace it with a real point you remember from the talk.

## 0:00–0:40 | Slide 1: The problem

“Hi, my project is a small video-language model prototype for finding when a person appears in a video. If you are looking for one person or one action, you may have to scrub through the clip several times. That becomes harder when the person is small or partly hidden.

“My question is: can a Video-LLM find an approximate time range when a person is visible, and give a reviewer enough information to check the result? The thermal image is from an earlier computer vision assignment. It illustrates the person-finding challenge; it is not the video used in my demo.”

## 0:40–1:25 | Slide 2: Connection to Keynote 6

“I chose Keynote 6, Fahad Khan’s ‘Towards Detailed Video Understanding in Generative AI Era.’ The talk’s focus on detailed video understanding inspired me to explore how a language question could help locate visual content over time. I narrowed that broad idea to one practical question: when is a person clearly visible in a short clip?

“My takeaway from watching the keynote was: **[add one specific idea you remember from the talk, in your own words].**

“For the prototype, I chose Gemini 3.8 Flash because it accepts video input through an API. Gemini is my implementation choice; I’m not saying it is the model used in the keynote. The app sends a video and a question, then returns a proposed time range and a description for a person to review.”

## 1:25–1:55 | Slide 3: Prototype and feasibility

“The app is deployed in Streamlit. In my test, I uploaded a seven-second warehouse clip. Gemini returned 00:00 to 00:07 and described a worker checking the shelves and interacting with a tablet. The screenshot here is the actual result from the app.

“That confirms the end-to-end request returned a useful response. It does not tell me how accurate the timestamps are. To measure that, I would mark the first and last frames where the person appears, then compare those times with Gemini’s interval.”

## 1:55–2:40 | Live app demonstration

“Here is ClipCheck running in the browser. I’ll choose the short warehouse clip, confirm that I have permission to send it to Google for analysis, and select Analyze video.

“Gemini has finished analyzing it. The result says the person is visible from 00:00 to 00:07. It describes the worker looking at the shelves and using a tablet. I can replay the clip around these times to check whether the description and interval match what is visible.”

## 2:40–3:00 | Close

“This prototype shows how a natural-language question can return a time range and a description to help someone start reviewing a video. My next step would be to compare more clips against manually marked intervals. Thank you.”

## Before recording

- Replace the Keynote takeaway sentence with one point you actually remember from the talk.
- Open [the deployed app](https://csc8830-week5-clipcheck.streamlit.app) and confirm the result is visible before recording. If you need to run the analysis again, leave time for Gemini to respond.
- Confirm permission before sending the clip to Google. Do not claim a numeric timestamp error unless you have manually marked the clip and calculated it.
- Keep the app result visible while speaking. If the live request takes too long, show the successful result already captured in the app and describe it accurately.
