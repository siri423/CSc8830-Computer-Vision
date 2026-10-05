from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

MIN_VIDEO_SECONDS = 30.0
FLOW_SECONDS = 30.0
MAX_WIDTH = 720


def _temp_video(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as f:
        f.write(data)
        return f.name


def _resize(frame: np.ndarray, width: int = MAX_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, width / max(w, h))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
    """Bilinearly interpolate a grayscale intensity at subpixel (x, y)."""
    h, w = image.shape[:2]
    if h < 2 or w < 2 or not (0 <= x <= w - 1 and 0 <= y <= h - 1):
        raise ValueError("The subpixel coordinate must be inside an image at least two pixels wide and high.")
    # At the final row or column, use the last valid pixel neighborhood and
    # let alpha or beta reach 1. This handles tracked points on image borders.
    x0, y0 = min(int(np.floor(x)), w - 2), min(int(np.floor(y)), h - 2)
    a, b = x - x0, y - y0
    p00, p10 = float(image[y0, x0]), float(image[y0, x0 + 1])
    p01, p11 = float(image[y0 + 1, x0]), float(image[y0 + 1, x0 + 1])
    return (1-a)*(1-b)*p00 + a*(1-b)*p10 + (1-a)*b*p01 + a*b*p11


def video_info(video_bytes: bytes, suffix: str) -> dict:
    path = _temp_video(video_bytes, suffix)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not read this file. Try exporting it as an H.264 MP4.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or count <= 1 or width <= 1 or height <= 1:
            raise ValueError("The video is missing usable frame-rate, size, or frame-count metadata.")
        return {"duration": count / fps, "fps": fps, "frames": count, "width": width, "height": height}
    finally:
        cap.release()
        Path(path).unlink(missing_ok=True)


def _flow_colour(flow: np.ndarray, scale: float) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = np.uint8(ang * 90 / np.pi)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.clip(mag / max(scale, 0.5) * 255, 0, 255))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dense_flow_video(video_bytes: bytes, suffix: str) -> tuple[bytes, dict]:
    """Render a side-by-side input and Farneback flow video for the first 30 s."""
    input_path = _temp_video(video_bytes, suffix)
    output_path = None
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            raise ValueError("The video could not be opened.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        if duration < MIN_VIDEO_SECONDS:
            raise ValueError(f"This clip is {duration:.1f} s. The assignment requires at least 30 s per video.")
        stride = max(1, round(fps / 10.0))
        out_fps = max(1.0, fps / stride)
        ok, first = cap.read()
        if not ok:
            raise ValueError("No readable frames were found.")
        first = _resize(first)
        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        h, w = first.shape[:2]
        output_file = tempfile.NamedTemporaryFile(suffix="_flow.mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (2*w, h))
        if not writer.isOpened():
            raise ValueError("OpenCV could not create an MP4 output on this system.")
        writer.write(np.hstack([first, np.zeros_like(first)]))
        sampled = 1
        magnitudes = []
        active_magnitudes = []
        frame_index = 1
        max_frame = min(total, int(round(FLOW_SECONDS * fps)))
        while frame_index < max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frame = _resize(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude = np.linalg.norm(flow, axis=2)
                scale = float(np.percentile(magnitude, 95)) if magnitude.size else 1.0
                coloured = _flow_colour(flow, scale)
                for y in range(20, h, 40):
                    for x in range(20, w, 40):
                        dx, dy = flow[y, x]
                        cv2.arrowedLine(coloured, (x, y), (int(x + dx), int(y + dy)), (255, 255, 255), 1, tipLength=0.25)
                cv2.putText(frame, "Input video", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 240, 240), 2)
                cv2.putText(coloured, "Farneback optical flow", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                writer.write(np.hstack([frame, coloured]))
                if magnitude.size:
                    magnitudes.append(float(np.median(magnitude)))
                    active_magnitudes.append(float(np.percentile(magnitude, 90)))
                prev_gray = gray
                sampled += 1
            frame_index += 1
        writer.release()
        with open(output_path, "rb") as f:
            result = f.read()
        return result, {"duration_seconds": duration, "processed_frames": sampled, "output_fps": out_fps,
                        "median_flow_pixels_per_sample": float(np.median(magnitudes)) if magnitudes else 0.0,
                        "mean_sampled_median_flow": float(np.mean(magnitudes)) if magnitudes else 0.0,
                        "mean_p90_flow_pixels_per_sample": float(np.mean(active_magnitudes)) if active_magnitudes else 0.0}
    finally:
        cap.release()
        Path(input_path).unlink(missing_ok=True)
        if output_path:
            Path(output_path).unlink(missing_ok=True)


def track_consecutive_pair(video_bytes: bytes, suffix: str, time_seconds: float = 1.0) -> tuple[np.ndarray, list[dict], dict]:
    """Track Shi-Tomasi points to the next distinct image with pyramidal Lucas-Kanade."""
    path = _temp_video(video_bytes, suffix)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("The video could not be opened.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total < 2:
            raise ValueError("This clip does not contain a usable pair of frames.")
        first_index = min(max(0, int(time_seconds * fps)), total - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_index)
        ok1, first = cap.read()
        if not ok1:
            raise ValueError("Could not read the selected frame.")
        # Screen captures of 30 fps material are often stored at 60 fps with
        # every other frame repeated. Skip exact or near-exact duplicates so
        # the LK check compares the next distinct image, not a duplicate.
        second = None
        second_index = first_index
        first_small = cv2.cvtColor(_resize(first), cv2.COLOR_BGR2GRAY)
        for candidate_index in range(first_index + 1, min(total, first_index + max(10, int(round(fps / 2))))):
            ok, candidate = cap.read()
            if not ok:
                break
            candidate_small = cv2.cvtColor(_resize(candidate), cv2.COLOR_BGR2GRAY)
            if float(cv2.absdiff(first_small, candidate_small).mean()) > 0.1:
                second, second_index = candidate, candidate_index
                break
        if second is None:
            raise ValueError("Could not find a distinct second frame after the selected time.")
        first, second = _resize(first), _resize(second)
        gray1 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(gray1, maxCorners=80, qualityLevel=.01, minDistance=12, blockSize=7)
        if points is None or len(points) < 4:
            raise ValueError("Too few trackable corners were detected in the first frame. Choose a more textured interval.")
        next_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if next_points is None or status is None:
            raise ValueError("Lucas-Kanade did not return tracked points for this frame pair.")
        p0 = points.reshape(-1, 2); p1 = next_points.reshape(-1, 2)
        valid = status.ravel().astype(bool) & np.isfinite(p1).all(axis=1)
        valid &= (p1[:, 0] >= 0) & (p1[:, 0] <= gray2.shape[1]-1)
        valid &= (p1[:, 1] >= 0) & (p1[:, 1] <= gray2.shape[0]-1)
        canvas = np.hstack([first.copy(), second.copy()])
        rows = []
        drawable = []
        for i, (a, b, good) in enumerate(zip(p0, p1, valid)):
            if not good:
                continue
            x0, y0 = map(float, a); x1, y1 = map(float, b)
            dx, dy = x1-x0, y1-y0
            track_id=len(rows)+1
            drawable.append((float(np.hypot(dx,dy)),track_id,x0,y0,x1,y1,dx,dy))
            rows.append({"track_id": track_id, "x0_px": x0, "y0_px": y0, "x1_px": x1, "y1_px": y1,
                         "dx_px": dx, "dy_px": dy, "displacement_px": float(np.hypot(dx,dy)),
                         "reference_x1_px": "", "reference_y1_px": ""})
        # Keep the side-by-side figure legible: draw the 30 tracks with the
        # largest measured motion, while retaining every valid track in the CSV.
        for _,track_id,x0,y0,x1,y1,dx,dy in sorted(drawable,reverse=True)[:30]:
            cv2.arrowedLine(canvas,(round(x0),round(y0)),(round(x0+dx),round(y0+dy)),(60,230,80),1,tipLength=.35)
            cv2.circle(canvas,(round(x0),round(y0)),3,(40,220,255),-1)
            cv2.circle(canvas,(round(x1+first.shape[1]),round(y1)),3,(40,220,255),-1)
            cv2.putText(canvas,str(track_id),(round(x0+4),round(y0-4)),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            cv2.putText(canvas,str(track_id),(round(x1+first.shape[1]+4),round(y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        if not rows:
            raise ValueError("No points remained valid between the two frames.")
        cv2.putText(canvas, "Frame A (t)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        cv2.putText(canvas, "Frame B (t + 1 frame)", (first.shape[1]+12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        # One actual subpixel sample for the bilinear-interpolation walkthrough.
        row = rows[0]
        bilinear_value = bilinear_sample(gray2, row["x1_px"], row["y1_px"])
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), rows, {"frame_index": first_index, "second_frame_index": second_index, "frame_gap": second_index-first_index, "fps": fps,
                 "frame_time_seconds": first_index / fps, "second_frame_time_seconds": second_index / fps, "bilinear_gray_at_tracked_point": bilinear_value,
                 "tracked_points": len(rows)}
    finally:
        cap.release()
        Path(path).unlink(missing_ok=True)


def four_view_planar_sfm(image_bytes: list[bytes], filenames: list[str], camera_matrix: np.ndarray,
                         calibration_width: int, calibration_height: int,
                         board_cols: int, board_rows: int, square_mm: float,
                         distortion: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Calibrated planar four-view reconstruction from detected checkerboard corners.

    Estimates each view pose with solvePnP against the measured checkerboard grid, then
    triangulates corresponding corners from adjacent views and reports camera centers.
    The known grid provides the metric reference and an evaluation target for SfM output.
    """
    if len(image_bytes) != 4 or len(filenames) != 4:
        raise ValueError("Upload exactly four views of the same textured planar object.")
    obj = np.zeros((board_cols * board_rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * float(square_mm)
    image_points=[]; image_shapes=[]; display=[]
    pattern=(board_cols, board_rows)
    for raw, name in zip(image_bytes, filenames):
        image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {name}.")
        h,w=image.shape[:2]; image_shapes.append((w,h))
        ok,corners=cv2.findChessboardCornersSB(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),pattern)
        if not ok:
            raise ValueError(f"Could not detect a {board_cols}x{board_rows} checkerboard in {name}.")
        corners=corners.reshape(-1,2).astype(np.float32)
        image_points.append(corners)
        vis=image.copy(); cv2.drawChessboardCorners(vis,pattern,corners.reshape(-1,1,2),True); display.append(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
    poses=[]; cameras=[]
    distortion = np.zeros((5, 1), dtype=np.float64) if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    for corners,(w,h) in zip(image_points,image_shapes):
        # Scale calibration intrinsics to each image's actual pixel dimensions.
        K=camera_matrix.copy().astype(np.float64)
        K[0,:] *= w / calibration_width
        K[1,:] *= h / calibration_height
        # Calibration distortion is specified in normalized camera coordinates and
        # remains valid when the intrinsic matrix is scaled for a resized image.
        ok,rvec,tvec=cv2.solvePnP(obj,corners,K,distortion,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: raise ValueError("Camera pose estimation failed for one checkerboard view.")
        R,_=cv2.Rodrigues(rvec); poses.append((R,tvec.reshape(3,1),K))
        center=(-R.T@tvec).reshape(3); cameras.append(center)
    all_points=[]; pair_stats=[]
    for i in range(3):
        R1,t1,K=poses[i]; R2,t2,_=poses[i+1]
        # Undistort pixel coordinates before linear triangulation.
        pts1=cv2.undistortPoints(image_points[i].reshape(-1,1,2),K,distortion).reshape(-1,2).T
        pts2=cv2.undistortPoints(image_points[i+1].reshape(-1,1,2),K,distortion).reshape(-1,2).T
        P1=np.hstack([R1,t1]); P2=np.hstack([R2,t2])
        homog=cv2.triangulatePoints(P1,P2,pts1,pts2)
        pts=(homog[:3]/np.maximum(homog[3],1e-9)).T
        good=np.isfinite(pts).all(axis=1) & (homog[3] > 1e-8)
        all_points.extend(pts[good])
        pair_stats.append({"view_pair":f"{i+1}-{i+2}","corners":int(good.sum())})
    return np.asarray(all_points,dtype=float), np.asarray(cameras,dtype=float), pair_stats
