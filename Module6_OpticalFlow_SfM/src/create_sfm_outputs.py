"""Reproduce the included calibrated four-view planar reconstruction outputs."""
from __future__ import annotations
import csv, json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.vision import four_view_planar_sfm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sfm_views'
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
config=json.loads((DATA/'calibration.json').read_text())
files=sorted(DATA.glob('view_*.jpeg'))
raw=[p.read_bytes() for p in files]
K=np.asarray(config['camera_matrix'],float)
dist=np.asarray(config['dist_coeffs'],float)
cols,rows=config['pattern_size_internal_corners']
points,cameras,pairs=four_view_planar_sfm(raw,[p.name for p in files],K,*config['image_size'],cols,rows,config['square_size_mm'],dist)
with (OUT/'triangulated_points.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['X_mm','Y_mm','Z_mm']); w.writerows(points)
with (OUT/'camera_centres.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['view','image','camera_x_mm','camera_y_mm','camera_z_mm'])
 for i,(name,c) in enumerate(zip(files,cameras),1): w.writerow([i,name.name,*c])
# reconstruct expected corners and compute grid-boundary estimate / flatness
expected_width=(cols-1)*config['square_size_mm']; expected_height=(rows-1)*config['square_size_mm']
result={
 'images':[p.name for p in files], 'detected_grid':[cols,rows], 'square_size_mm':config['square_size_mm'],
 'calibration_reprojection_rms_px':config['reprojection_rms_px'], 'triangulated_points':len(points),
 'triangulated_point_count_per_adjacent_pair':pairs,
 'expected_planar_boundary_mm':[expected_width,expected_height],
 'triangulated_bounds_mm':{'x':[float(points[:,0].min()),float(points[:,0].max())], 'y':[float(points[:,1].min()),float(points[:,1].max())], 'z':[float(points[:,2].min()),float(points[:,2].max())]},
 'boundary_extent_mm':[float(np.ptp(points[:,0])),float(np.ptp(points[:,1]))],
 'extent_error_mm':[float(np.ptp(points[:,0])-expected_width),float(np.ptp(points[:,1])-expected_height)],
 'z_range_mm':float(np.ptp(points[:,2])), 'method':'Calibrated planar reconstruction: solvePnP poses from known grid; undistort matched corners and triangulate adjacent views.'}
(OUT/'sfm_summary.json').write_text(json.dumps(result,indent=2))
fig=plt.figure(figsize=(8.2,5.2)); ax=fig.add_subplot(111,projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2],s=18,label='Triangulated checkerboard corners')
ax.scatter(cameras[:,0],cameras[:,1],cameras[:,2],c='crimson',marker='^',s=70,label='Camera centres')
for i,p in enumerate(cameras): ax.text(*p,f' C{i+1}',fontsize=8)
ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)');ax.set_title('Four-view calibrated planar reconstruction');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'four_view_reconstruction.png',dpi=160);plt.close(fig)
print(json.dumps(result,indent=2))
