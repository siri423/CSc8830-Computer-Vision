# Module 6: Optical Flow, Tracking, and Structure from Motion

This project demonstrates dense optical flow and two-frame point tracking on two videos, then estimates camera poses and triangulates a planar checkerboard from four views. It is built around the assignment's required motion evidence and multi-view geometry.

## Live demo

[Open the Module 6 Streamlit app](https://csc8830-module6-optical-flow-sfm.streamlit.app).

## Run the demo

Use Python 3.10 or newer. From this folder:

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload two separate videos, each at least 30 seconds long and containing visible motion. The app samples the first 30 seconds for a side-by-side Farneback flow visualization. It tracks Shi-Tomasi corners between adjacent distinct images with pyramidal Lucas-Kanade and exports their pixel coordinates. Repeated frames are skipped when the source capture contains duplicates. The output also reports a bilinear grayscale sample at a subpixel tracked location.

For the multi-view part, use the four included images in `data/sfm_views/` by default; the app also accepts four replacement images. They show a known 9 by 6 internal-corner checkerboard photographed with the same phone and camera mode as the calibration in `data/sfm_views/calibration.json`. The code uses the saved intrinsics and distortion coefficients, estimates each view pose using the known 25 mm grid, and triangulates corresponding points. This is a calibrated planar reconstruction with known geometry, an educational feasibility demonstration rather than unconstrained SfM of an unknown object.

Recreate the included four-view tables and plot with `python -m src.create_sfm_outputs`. Rebuild the PDF report with `python build_report.py`.


## Method and validation

Brightness constancy gives `Ix*u + Iy*v + It = 0`; the aperture problem is handled locally by the Lucas-Kanade least-squares estimate. Dense Farneback flow supplies a motion field for visualization. For two-frame validation, inspect a numbered feature in the annotated frame pair and compare its manually read location with the exported `(x1_px, y1_px)`. The report records two approximate Frame B coordinate checks and their Euclidean endpoint errors; these visual readings are not independent ground truth. Bilinear interpolation uses the four neighboring pixel intensities and the tracked subpixel coordinate.

The calibration was previously estimated from the same phone images with 0.722 px reprojection RMS. Four included views are real calibration images, not synthetic renders. Camera locations are expressed in the checkerboard coordinate system; the known checkerboard provides the metric scale and expected planar boundary.

## Video analysis evidence

Two continuous motion clips were analyzed locally for this assignment: a 35.83-second unpacking and assembly scene and a 45.03-second road-traffic scene. The first 30 seconds of each clip were used to create the dense-flow visualizations. The upload-ready copies are each under the repository app limit of 20 MB. Source recordings and generated flow videos remain in the local working folder and are not included in this public repository. The numeric summary records the processing durations, track counts, pixel displacements, and bilinear samples. The tracker skips repeated images in the 60 fps captures before estimating motion between distinct frames.

## References

- Szeliski, R. *Computer Vision: Algorithms and Applications*, 2nd ed., sections on optical flow and structure from motion.
- OpenCV documentation: `calcOpticalFlowFarneback`, `calcOpticalFlowPyrLK`, `findChessboardCornersSB`, `solvePnP`, and `triangulatePoints`.
- Course lecture playlist: [Optical Flow | Structure from Motion | Object Tracking](https://www.youtube.com/playlist?list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV).
