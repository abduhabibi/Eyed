# extract_3d_path.py
# Feature 3: 3D trajectory of medial canthus (inner eye corner).
# Normalized by interocular distance for scale invariance.
# Compensates for small head movements by using relative coordinates.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean

INPUT_VIDEO_PATH = "squeeze_video.avi"
MEDIAL_CANTHUS = 133   # right eye inner corner
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
if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

traj_norm = []   # normalized (x/ref, y/ref, z/ref)
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
        pt = landmarks[MEDIAL_CANTHUS]
        # Normalize coordinates by interocular distance (makes scale invariant)
        x_norm = pt.x / ref_dist
        y_norm = pt.y / ref_dist
        z_norm = pt.z / ref_dist
        traj_norm.append([x_norm, y_norm, z_norm])
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
cap.release()

traj_norm = np.array(traj_norm)
if len(traj_norm) < SG_WINDOW:
    print("Not enough frames")
    exit()

# Smooth
traj_smooth = np.zeros_like(traj_norm)
for i in range(3):
    traj_smooth[:, i] = savgol_filter(traj_norm[:, i], SG_WINDOW, SG_ORDER)

# Path length
path_len = sum(euclidean(traj_smooth[i], traj_smooth[i-1]) for i in range(1, len(traj_smooth)))
# Max displacement from start
max_disp = np.max(np.linalg.norm(traj_smooth - traj_smooth[0], axis=1))

# Resample to fixed length (100 points) for AI
target_len = 100
resampled = np.zeros((target_len, 3))
for i in range(3):
    resampled[:, i] = np.interp(np.linspace(0, len(traj_smooth)-1, target_len),
                                np.arange(len(traj_smooth)), traj_smooth[:, i])
trajectory_feature = resampled.flatten()   # 300 numbers

feature_vector = np.array([path_len, max_disp] + list(trajectory_feature[:20]))  # first 20 for brevity

print("\n=== Feature 3: 3D Path ===")
print(f"Path length: {path_len:.4f}, Max displacement: {max_disp:.4f}")
print(f"Feature vector length: {len(feature_vector)}")
np.save("feature3_3d_path.npy", feature_vector)
