"""Assignment 6 demo: dense optical flow, two-frame tracking, and planar multi-view reconstruction."""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.vision import dense_flow_video, four_view_planar_sfm, track_consecutive_pair, video_info

st.set_page_config(page_title="Assignment 6 | Motion and SfM", layout="wide")
st.title("Assignment 6: Motion and Structure from Motion")
st.caption("Dense optical flow, two-frame Lucas-Kanade tracking, and four-view planar reconstruction")
st.info("Use two videos of at least 30 seconds each. The optical-flow videos below show the input and its computed motion field side by side.")

with st.expander("Equations used in this demonstration", expanded=False):
    st.markdown(r"""
**Brightness constancy:** a moving image point keeps approximately the same intensity:
\(I(x,y,t)=I(x+u\Delta t,y+v\Delta t,t+\Delta t)\).
Using a first-order Taylor expansion gives the optical-flow constraint
\(I_xu+I_yv+I_t=0\). One equation has two unknowns, so Lucas-Kanade assumes nearby points have similar motion and solves a local least-squares system
\(\mathbf d=(A^TA)^{-1}A^T\mathbf b\), where \(\mathbf d=[u,v]^T\).

**Bilinear interpolation:** for a subpixel point \((x_0+\alpha,y_0+\beta)\), interpolate the four neighbouring pixels:
\(I=(1-\alpha)(1-\beta)I_{00}+\alpha(1-\beta)I_{10}+(1-\alpha)\beta I_{01}+\alpha\beta I_{11}\).
The code computes this sample at a tracked point in the second frame.

**Multi-view geometry:** image point \(\tilde{\mathbf x}_i\) and 3D point \(\tilde{\mathbf X}\) satisfy
\(\tilde{\mathbf x}_i\sim K[R_i\mid t_i]\tilde{\mathbf X}\). Corresponding corners from two views are triangulated; the known checkerboard grid gives a metric reference for checking the reconstruction.
""")

st.header("A. Optical flow and tracking")
tabs = st.tabs(["Video 1", "Video 2", "Four-view SfM"])
for number, tab in enumerate(tabs[:2], start=1):
    with tab:
        up = st.file_uploader(f"Choose video {number}", type=["mp4", "mov", "avi", "mkv", "webm"], key=f"video_{number}", help="Use a video under 20 MB. The prepared MP4 copies fit the upload limit.")
        if up:
            raw = up.getvalue(); suffix = "." + up.name.rsplit(".",1)[-1].lower()
            try:
                info = video_info(raw, suffix)
                st.write(f"{info['duration']:.1f} s, {info['width']} by {info['height']} pixels, {info['fps']:.2f} fps")
                if info["duration"] < 30:
                    st.error("The assignment requires at least 30 seconds of motion in each video. This file is too short.")
                else:
                    last_pair_time = max(0.0, info["duration"] - 1.0 / info["fps"])
                    tracking_time = st.slider("Choose when to check a frame and the next distinct image", 0.0, float(last_pair_time),
                                               min(1.0, float(last_pair_time)), 0.5, key=f"tracking_time_{number}")
                if info["duration"] >= 30 and st.button(f"Analyze video {number}", type="primary", key=f"run_{number}"):
                    with st.spinner("Computing Farneback flow and tracking distinct frames…"):
                        flow_bytes, flow_stats = dense_flow_video(raw, suffix)
                        pair_img, tracks, pair_stats = track_consecutive_pair(raw, suffix, tracking_time)
                    st.session_state[f"flow_{number}"] = (flow_bytes, flow_stats)
                    st.session_state[f"pair_{number}"] = (pair_img, tracks, pair_stats)
            except Exception as exc:
                st.error(str(exc))
        flow_data = st.session_state.get(f"flow_{number}")
        pair_data = st.session_state.get(f"pair_{number}")
        if flow_data:
            flow_bytes, stats = flow_data
            st.subheader("Optical-flow visualization")
            st.video(flow_bytes)
            st.caption(f"Farneback dense flow; {stats['processed_frames']} sampled frames; average 90th-percentile motion {stats['mean_p90_flow_pixels_per_sample']:.2f} px per sampled step. The median also includes stationary background pixels.")
            if stats['mean_p90_flow_pixels_per_sample'] < 0.05:
                st.warning("Very little motion was estimated in most of the frame. Try a clip with a larger or more textured moving subject.")
            st.download_button("Download optical-flow video", flow_bytes, f"video{number}_optical_flow.mp4", "video/mp4", key=f"download_flow_{number}")
        if pair_data:
            pair_img, tracks, pair_stats = pair_data
            st.subheader("Tracking between two distinct frames")
            st.image(pair_img, caption=f"Lucas-Kanade tracks from {pair_stats['frame_time_seconds']:.3f}s to {pair_stats['second_frame_time_seconds']:.3f}s. Repeated identical frames are skipped when present.", use_container_width=True)
            df = pd.DataFrame(tracks)
            st.dataframe(df[["track_id","x0_px","y0_px","x1_px","y1_px","dx_px","dy_px","displacement_px"]].round(2), use_container_width=True, hide_index=True)
            st.caption(f"Bilinear grayscale sample at the first tracked subpixel location: {pair_stats['bilinear_gray_at_tracked_point']:.2f} on a 0 to 255 scale.")
            track_id = st.selectbox("Choose a track to validate by eye", df.track_id.tolist(), key=f"manual_track_{number}")
            selected = df.loc[df.track_id == track_id].iloc[0]
            mx, my = st.columns(2)
            observed_x = mx.number_input("Observed x in frame B (pixels)", value=float(selected.x1_px), key=f"obs_x_{number}")
            observed_y = my.number_input("Observed y in frame B (pixels)", value=float(selected.y1_px), key=f"obs_y_{number}")
            endpoint_error = float(np.hypot(observed_x-selected.x1_px, observed_y-selected.y1_px))
            st.write(f"Endpoint error against the manually entered location: **{endpoint_error:.2f} px**. Set the observed coordinates by zooming into the annotated pair and reading the pixel location; the default equals the prediction and is not an independent measurement.")
            st.download_button("Download tracked pixel coordinates", df.to_csv(index=False).encode(), f"video{number}_tracking.csv", "text/csv", key=f"download_tracks_{number}")

with tabs[2]:
    st.header("B. Four views of one planar target")
    st.write("A set of four real checkerboard views and the phone calibration from the earlier camera-calibration work are included. Run these directly, or upload four replacement views captured with the same calibrated camera mode.")
    use_custom = st.checkbox("Use my own four images instead of the included views")
    pics = st.file_uploader("Upload exactly four viewpoint images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="sfm_images")
    c1,c2,c3 = st.columns(3)
    fx=c1.number_input("fx (pixels)", min_value=1.0, value=3095.07)
    fy=c2.number_input("fy (pixels)", min_value=1.0, value=3094.48)
    cx=c3.number_input("cx (pixels)", min_value=0.0, value=932.21)
    c4,c5,c6=c1,c2,c3
    cy=c4.number_input("cy (pixels)", min_value=0.0, value=2024.67)
    cal_w=c5.number_input("Calibration image width", min_value=1, value=1884)
    cal_h=c6.number_input("Calibration image height", min_value=1, value=4080)
    b1,b2,b3=st.columns(3)
    cols=b1.number_input("Internal checkerboard corners across", min_value=2, max_value=30, value=9)
    rows=b2.number_input("Internal checkerboard corners down", min_value=2, max_value=30, value=6)
    square=b3.number_input("Square size (mm)", min_value=1.0, value=25.0)
    if use_custom and pics and len(pics)!=4: st.warning(f"Select exactly four images of the same board; selected {len(pics)}.")
    run_reconstruction = st.button("Reconstruct the four views", type="primary")
    if run_reconstruction and (use_custom and pics and len(pics)==4 or not use_custom):
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=float)
        try:
            if use_custom:
                image_data=[p.getvalue() for p in pics]; names=[p.name for p in pics]
            else:
                sample_dir=Path(__file__).parent/"data"/"sfm_views"
                sample_files=sorted(sample_dir.glob("view_*.jpeg"))
                image_data=[p.read_bytes() for p in sample_files]; names=[p.name for p in sample_files]
            with st.spinner("Detecting board corners, estimating camera poses, and triangulating…"):
                cfg=json.loads((Path(__file__).parent/"data"/"sfm_views"/"calibration.json").read_text())
                distortion=np.asarray(cfg["dist_coeffs"],dtype=float)
                pts,cameras,pairs=four_view_planar_sfm(image_data,names,K,int(cal_w),int(cal_h),int(cols),int(rows),float(square),distortion)
            st.session_state["sfm_result"]=(pts,cameras,pairs,names)
        except Exception as exc: st.error(str(exc))
    result=st.session_state.get("sfm_result")
    if result:
        pts,cameras,pairs,names=result
        fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111,projection="3d")
        if len(pts): ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=10,label="Triangulated grid points")
        ax.scatter(cameras[:,0],cameras[:,1],cameras[:,2],c="red",marker="^",s=50,label="Camera centres")
        for i,p in enumerate(cameras): ax.text(*p,f" C{i+1}")
        ax.set_xlabel("X (mm)");ax.set_ylabel("Y (mm)");ax.set_zlabel("Z (mm)");ax.legend();st.pyplot(fig);plt.close(fig)
        st.dataframe(pd.DataFrame(cameras,columns=["camera_x_mm","camera_y_mm","camera_z_mm"]).round(2),hide_index=True)
        st.caption("The checkerboard's known square spacing sets the metric reference. These camera poses and triangulated points demonstrate reconstruction of a calibrated planar target.")
        st.download_button("Download reconstructed 3D points",pd.DataFrame(pts,columns=["X_mm","Y_mm","Z_mm"]).to_csv(index=False).encode(),"sfm_points.csv","text/csv")

st.divider()
st.caption("For the submission: use two qualifying 30-second videos, include results from each, show a manual pixel-coordinate check, and document the four-view camera setup and parameters in the report.")
