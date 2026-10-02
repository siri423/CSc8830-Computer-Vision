# Before submitting

1. Watch Keynote 6 in class: Fahad Khan, “Towards Detailed Video Understanding in Generative AI Era.” Write down one point from the talk in your own words. Add that personal takeaway to your presentation notes or use it when presenting Slide 2.

2. Create a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey). Keep the key private. Do not paste it into chat, source code, GitHub, or the slide deck.

3. The app has returned a successful result for a seven-second warehouse clip: Gemini reported 00:00–00:07 and described a worker checking shelves and using a tablet. The manual reference interval and boundary errors have not been measured, so do not present numeric accuracy claims.

4. The app is deployed at [csc8830-week5-clipcheck.streamlit.app](https://csc8830-week5-clipcheck.streamlit.app). In its Streamlit Community Cloud settings, open **Secrets** and add this secret:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

   Keep the secret in Streamlit’s settings only. Do not add it to the repository. The app will restart after the secret saves.

5. Use `output/Week5_Challenge1_Presentation_v5.pptx` for the submission. Read `recording_script.md` while preparing your three-minute screen recording. Keep the deck to three slides. Add one personal takeaway from the keynote after watching it in class.

6. Upload the PowerPoint file to the Week 5 Challenge assignment in Google Classroom before the deadline. The GitHub repository is [CSc8830-Computer-Vision](https://github.com/siri423/CSc8830-Computer-Vision).

The project summary and feasibility worksheet are in [`REPORT.md`](REPORT.md).
