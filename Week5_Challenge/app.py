"""A small Streamlit prototype for finding a person in a short video."""
from __future__ import annotations

import os
import hashlib
import tempfile
import time
from pathlib import Path

import streamlit as st
from google import genai

MODEL_ID = "gemini-3.8-flash"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FILE_READY_TIMEOUT_SECONDS = 180
FILE_POLL_INTERVAL_SECONDS = 3
MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".3gp": "video/3gpp",
}
DEFAULT_QUESTION = (
    "Find the first and last time the main person is clearly visible. "
    "Give approximate start and end timestamps in MM:SS format. Describe "
    "what the person is doing and the visual evidence. If no person is "
    "clearly visible, say so. Do not guess details that are not visible."
)


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets or the local environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return str(key).strip() if key else None


def explain_api_error(error: Exception) -> str:
    """Turn common Gemini service errors into clear next steps."""
    detail = str(error).lower()
    if "prepayment" in detail or "payment_required" in detail or "402" in detail:
        return (
            "Gemini rejected the request because the linked project has no usable "
            "prepaid API credits. Check the project’s Billing page in Google AI "
            "Studio. Add credits if you want to use this paid project, or choose "
            "a project and model with an available free tier. A new key for the "
            "same project will not restore its balance."
        )
    if "429" in detail or "rate_limit" in detail or "resource_exhausted" in detail:
        return (
            "Gemini reported a usage limit for this project. Check its limits in "
            "Google AI Studio and try again after the limit resets, or use a "
            "shorter clip."
        )
    if "401" in detail or "unauthenticated" in detail or "invalid api key" in detail:
        return (
            "Gemini did not accept the API key. Check that GEMINI_API_KEY in "
            "Streamlit Secrets belongs to a project with Gemini API access."
        )
    if "403" in detail or "permission_denied" in detail:
        return (
            "Gemini denied access to this model or project. Check the model and "
            "API access settings in Google AI Studio."
        )
    return (
        "Gemini could not complete the request. Check the model access and "
        "service status, then try a short MP4 or MOV clip."
    )


def file_state_name(file_obj) -> str:
    state = getattr(file_obj, "state", None)
    return str(getattr(state, "name", state or "")).upper()


def wait_until_active(client, uploaded_file, progress=None):
    """Poll Gemini's Files API with a timeout and useful failure messages."""
    deadline = time.monotonic() + FILE_READY_TIMEOUT_SECONDS
    current = uploaded_file
    while True:
        state = file_state_name(current)
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError("Google could not process this video. Try a short MP4 or MOV clip.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Video processing took too long. Try a shorter or smaller clip.")
        if progress:
            progress("Google is preparing the video…")
        time.sleep(FILE_POLL_INTERVAL_SECONDS)
        current = client.files.get(name=current.name)


def analyze_video(video_bytes: bytes, filename: str, question: str, api_key: str) -> str:
    """Upload one video, ask Gemini a question, and delete the uploaded file."""
    suffix = Path(filename).suffix.lower()
    mime_type = MIME_TYPES.get(suffix)
    if not mime_type:
        raise ValueError("Choose an MP4, MOV, AVI, WebM, MPEG or 3GP video.")
    if not video_bytes:
        raise ValueError("The selected file is empty. Choose another video.")
    if len(video_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("This prototype accepts videos up to 20 MB.")
    if not question.strip():
        raise ValueError("Enter a question about the video.")

    client = genai.Client(api_key=api_key)
    remote_file = None
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        remote_file = client.files.upload(file=temp_path)
        remote_file = wait_until_active(client, remote_file)
        video_uri = getattr(remote_file, "uri", None)
        remote_name = getattr(remote_file, "name", None)
        remote_mime = getattr(remote_file, "mime_type", None) or mime_type
        if not video_uri or not remote_name:
            raise RuntimeError("Google did not return a usable video reference. Try again with another clip.")

        interaction = client.interactions.create(
            model=MODEL_ID,
            input=[
                {
                    "type": "video",
                    "uri": video_uri,
                    "mime_type": remote_mime,
                    "processing": "agentic",
                },
                {"type": "text", "text": question.strip()},
            ],
            store=False,
        )
        answer = getattr(interaction, "output_text", None)
        if not answer or not str(answer).strip():
            raise RuntimeError("Gemini returned an empty answer. Try a more specific question or another clip.")
        return str(answer).strip()
    finally:
        if remote_file is not None and getattr(remote_file, "name", None):
            try:
                client.files.delete(name=remote_file.name)
            except Exception:
                # The Files API also expires uploads automatically.
                pass
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


st.set_page_config(page_title="ClipCheck | Video understanding", layout="wide")
st.title("ClipCheck: find a person in a video")
st.write(
    "This small test asks a video model to find when a person appears and "
    "describe the visible evidence. It is inspired by detailed, time-grounded "
    "video understanding from Keynote 6."
)
st.info(
    "Model: Google Gemini 3.8 Flash. It returns a text answer with approximate "
    "timestamps, not a pixel mask or a verified ground-truth label."
)

with st.expander("Before you upload"):
    st.write(
        "The video is sent to Google Gemini for analysis. This prototype deletes "
        "the temporary upload after the answer. Google also automatically deletes "
        "Files API uploads after 48 hours. Use only a clip you have permission to "
        "share. Free-tier data may be used to improve Google products, so do not "
        "upload private or sensitive footage."
    )
    st.caption("The model samples video content and can miss brief actions or give an approximate timestamp. Check its answer against the clip.")

api_key = get_api_key()
if not api_key:
    st.warning("Add GEMINI_API_KEY in Streamlit Community Cloud settings or your local environment before running the model.")
    st.link_button("Get a Gemini API key in Google AI Studio", "https://aistudio.google.com/app/apikey")

upload = st.file_uploader(
    "Choose a short video",
    type=list(MIME_TYPES.keys()),
    help="Supported formats: MP4, MOV, AVI, WebM, MPEG and 3GP. Maximum file size: 20 MB.",
)
question = st.text_area("What should Gemini look for?", value=DEFAULT_QUESTION, height=110)
consent = st.checkbox("I have permission to send this video to Google for analysis.")

if upload is not None:
    video_bytes = upload.getvalue()
    current_video_id = hashlib.sha256(video_bytes).hexdigest()
    current_question_id = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    if upload.size == 0:
        st.error("The selected file is empty. Choose another video.")
    elif upload.size > MAX_UPLOAD_BYTES:
        st.error("This prototype accepts videos up to 20 MB. Choose a shorter or smaller clip.")
    else:
        st.video(video_bytes, format=MIME_TYPES.get(Path(upload.name).suffix.lower()))
        run_analysis = st.button(
            "Analyze video",
            type="primary",
            disabled=not (api_key and consent and question.strip()),
        )
        if run_analysis:
            try:
                with st.status("Sending the video to Gemini…", expanded=True) as status:
                    try:
                        answer = analyze_video(video_bytes, upload.name, question, api_key)
                    except Exception:
                        status.update(label="Gemini request did not complete", state="error", expanded=False)
                        raise
                    status.update(label="Analysis finished", state="complete", expanded=False)
                st.session_state["clipcheck_answer"] = answer
                st.session_state["clipcheck_filename"] = upload.name
                st.session_state["clipcheck_model"] = MODEL_ID
                st.session_state["clipcheck_video_id"] = current_video_id
                st.session_state["clipcheck_question_id"] = current_question_id
            except ValueError as error:
                st.error(str(error))
            except TimeoutError as error:
                st.error(str(error))
            except Exception as error:
                message = explain_api_error(error)
                st.error(message)
                if "prepaid API credits" in message:
                    st.link_button(
                        "Open Google AI Studio projects and billing",
                        "https://aistudio.google.com/projects",
                    )

if (
    upload is not None
    and st.session_state.get("clipcheck_answer")
    and st.session_state.get("clipcheck_video_id") == current_video_id
    and st.session_state.get("clipcheck_question_id") == current_question_id
):
    st.subheader("Gemini’s answer")
    st.caption(
        f"{st.session_state.get('clipcheck_filename', 'Video')}  |  "
        f"{st.session_state.get('clipcheck_model', MODEL_ID)}"
    )
    st.write(st.session_state["clipcheck_answer"])
    st.download_button(
        "Download this result",
        st.session_state["clipcheck_answer"],
        file_name="clipcheck_result.txt",
        mime="text/plain",
    )
    st.caption("Treat timestamps as estimates. Confirm them by replaying the video.")

st.divider()
st.caption("Feasibility check: compare the model’s time span with a manually reviewed reference on the same clip.")
