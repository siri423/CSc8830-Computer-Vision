# Before submitting

1. Watch Keynote 6 in class: Fahad Khan, “Towards Detailed Video Understanding in Generative AI Era.” Write down one point from the talk in your own words. Add that personal takeaway to your presentation notes or use it when presenting Slide 2.

2. Create a Gemini API key at [Google AI Studio](https://aistudio.google.com/app/apikey). Keep the key private. Do not paste it into chat, source code, GitHub, or the slide deck.

3. Run the app once with a short clip you are allowed to share. Follow the local setup in this folder’s README. Check the answer against the video. The first run should give you real output for the feasibility slide; do not invent a result if the API is unavailable.

4. For Streamlit Community Cloud, create a new app from `siri423/CSc8830-Computer-Vision`, branch `main`, with `Week5_Challenge/app.py` as the app file. In the app’s settings, add this secret:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

   Keep the secret in Streamlit’s settings only. Do not add it to the repository.

5. Open `output/Week5_Challenge1_VideoGrounding.pptx`. Update the feasibility slide with what happened when you ran the model, including the clip length and whether the approximate timestamps matched your own review. Keep the deck to three slides.

6. Upload the PowerPoint file to the Week 5 Challenge assignment in Google Classroom before the deadline. The GitHub repository is [CSc8830-Computer-Vision](https://github.com/siri423/CSc8830-Computer-Vision).
