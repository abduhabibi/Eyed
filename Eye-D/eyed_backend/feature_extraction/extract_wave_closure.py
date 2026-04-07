# extract_wave_closure.py
# Feature 5: Wave‑like closure pattern – segment delays.
# Normalized distances, but time delays are already absolute.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

INPUT_VIDEO_PATH = "squeeze_video.avi"
# Right eye upper lid segments (inner, middle, outer)
UPPER = [159, 160, 161]   # inner, middle, outer
LOWER = [145, 146, 147]
LEFT_CANTHUS = 133
RIGHT_CANTHUS = 362
SG_WINDOW = 15
SG_ORDER = 3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_interocular_distance(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left = landmarks[LEFT_CANTHUS]
    right = landmarks[RIGHT_CANTHUS]
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    return np.linalg.norm(left_pt - right_pt)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

seg_norm = [[] for _ in range(3)]   # inner, middle, outer
timestamps = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        ref_dist = get_interocular_distance(landmarks, frame.shape)
        if ref_dist < 1:
            continue
        h, w = frame.shape[:2]
        for i, (u, l) in enumerate(zip(UPPER, LOWER)):
            upper_y = landmarks[u].y * h
            lower_y = landmarks[l].y * h
            dist_px = lower_y - upper_y
            seg_norm[i].append(dist_px / ref_dist)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
cap.release()

if len(seg_norm[0]) < SG_WINDOW:
    print("Not enough frames")
    exit()

# Smooth and normalize each segment to [0,1] (1 = fully open)
smoothed = []
for i in range(3):
    smooth = savgol_filter(seg_norm[i], SG_WINDOW, SG_ORDER)
    max_val = np.max(smooth)
    min_val = np.min(smooth)
    if max_val - min_val > 0:
        smooth = (smooth - min_val) / (max_val - min_val)
    else:
        smooth = np.ones_like(smooth) * 0.5
    smoothed.append(smooth)

# Find 50% closure time for each segment
threshold = 0.5
closure_times = []
for i in range(3):
    idx = np.where(smoothed[i] <= threshold)[0]
    if len(idx) > 0:
        closure_times.append(timestamps[idx[0]])
    else:
        closure_times.append(np.nan)

# Delays
if not np.isnan(closure_times[0]) and not np.isnan(closure_times[1]):
    delay_inner_middle = closure_times[1] - closure_times[0]
else:
    delay_inner_middle = 0
if not np.isnan(closure_times[1]) and not np.isnan(closure_times[2]):
    delay_middle_outer = closure_times[2] - closure_times[1]
else:
    delay_middle_outer = 0

feature_vector = np.array([delay_inner_middle, delay_middle_outer])
print("\n=== Feature 5: Wave Closure ===")
print(f"Delay inner→middle (s): {delay_inner_middle:.4f}")
print(f"Delay middle→outer (s): {delay_middle_outer:.4f}")
np.save("feature5_wave_closure.npy", feature_vector)
