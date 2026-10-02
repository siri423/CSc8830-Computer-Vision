# Three-minute presentation and app demo

Keep the three slides on screen for the first part, then switch to the live app. The example result below is from the successful run shown in the app. Before recording, replay the full clip if you want to report a manual timestamp comparison.

## 0:00–0:45 | Slide 1: Problem

“Hello. This is a small Video-LLM prototype for finding when a person appears in a video. Reviewing a clip for one person can take several passes. First, someone has to find the person. Then they have to locate the moment they need. Small or partly hidden people make that search harder.

“The problem statement is: can a video language model identify an approximate time range when a person is visible and give a reviewer enough evidence to check the moment? The thermal image here is an example from an earlier computer vision assignment. It motivates the person-finding task.”

## 0:45–1:25 | Slide 2: Idea

“Keynote 6 by Fahad Khan is titled ‘Towards Detailed Video Understanding in Generative AI Era.’ The workshop description focuses on detailed video understanding and grounding video content from text queries. This project tests a narrower question: when is a person visible?

“The prototype uses Gemini 3.8 Flash through the Gemini API because it accepts video input. The model is our implementation choice; the keynote provides the inspiration. The app sends a clip and a question, then returns a proposed time range and a short explanation for a person to review.”

## 1:25–1:50 | Slide 3: Prototype and feasibility

"The app is deployed in Streamlit. I tested it with a seven-second warehouse clip. Gemini returned an interval from 00:00 to 00:07 and described a worker checking the shelves and using a tablet. That shows the video question and response flow works. To measure timestamp accuracy, I would mark the first and last visible frames myself and compare those times with Gemini’s answer. I have not recorded that manual interval yet, so I won’t claim a numeric timing error."

## 1:50–2:40 | Live app demonstration

“Here is the ClipCheck app. I’ll upload a short video, enter the question, and confirm that I have permission to send the clip to Google for analysis. I’ll select Analyze video and wait for Gemini’s response.”

**After the result appears, say:**

"For this seven-second clip, Gemini returned 00:00 to 00:07 and described a worker in a warehouse checking the inventory shelves and interacting with a tablet. The person is visible in the opening frames. A full manual timestamp comparison is still needed before I can report how close those boundaries are."

## 2:40–3:00 | Close

"This run shows that a video question can return a candidate time range and a description that a reviewer can check. A larger test with manually marked intervals would be needed to measure how consistently the timestamps match. Thank you."

## If the model call fails

Do not read the result paragraph above. Say: “The app is deployed, but this run did not return a model answer. I can show the prototype flow, but I cannot report a timestamp comparison until the API request succeeds.”

## Recording checklist

- Watch Keynote 6 in class and add one takeaway in your own words before presenting.
- Use a short MP4 or MOV clip under 20 MB. Choose a clip you have permission to share.
- Open [the deployed app](https://csc8830-week5-clipcheck.streamlit.app) before recording and make sure the API key is configured in Streamlit Secrets.
- Check the consent box and show the successful result. If you manually review the full clip, add the reference interval and calculate each boundary difference before recording.
- Keep the recording near three minutes. Leave enough time for the model to respond, or trim the upload/setup pause during editing.
