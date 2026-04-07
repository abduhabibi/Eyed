# extract_main_sequence.py
# Feature 2: Main Sequence – amplitude vs. peak velocity relationship.
# Requires multiple squeeze videos (or a single video with multiple squeezes).
# Normalized by interocular distance.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
import os
import glob

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
VIDEO_FOLDER = "squeeze_videos_person_A/"   # CHANGE THIS
UPPER_LID = 159
LOWER_LID = 145
LEFT_CANTHUS = 133
RIGHT_CANTHUS = 362
SG_WINDOW = 15
SG_ORDER = 3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def get_interocular_distance(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left = landmarks[LEFT_CANTHUS]
    right = landmarks[RIGHT_CANTHUS]
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    return np.linalg.norm(left_pt - right_pt)

def extract_amplitude_peak_velocity(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    norm_distances = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            ref_dist = get_interocular_distance(landmarks, frame.shape)
            if ref_dist < 1:
                continue
            upper_y = landmarks[UPPER_LID].y * h
            lower_y = landmarks[LOWER_LID].y * h
            dist_px = lower_y - upper_y
            norm_dist = dist_px / ref_dist
            norm_distances.append(norm_dist)
    cap.release()

    if len(norm_distances) < SG_WINDOW:
        return None, None

    dist_smooth = savgol_filter(norm_distances, SG_WINDOW, SG_ORDER)
    dt = 1.0 / fps
    velocity = np.gradient(dist_smooth) / dt
    amplitude = np.max(dist_smooth) - np.min(dist_smooth)
    peak_velocity = np.max(np.abs(velocity))
    return amplitude, peak_velocity

# -------------------------------
# 2. PROCESS ALL VIDEOS
# -------------------------------
video_paths = glob.glob(os.path.join(VIDEO_FOLDER, "*.[amv][pm4]*"))  # .avi, .mp4, .mov
amplitudes, peak_velocities = [], []
for path in video_paths:
    amp, vel = extract_amplitude_peak_velocity(path)
    if amp is not None and vel is not None:
        amplitudes.append(amp)
        peak_velocities.append(vel)

if len(amplitudes) < 3:
    print("Need at least 3 valid squeezes.")
    exit()

# Linear regression
slope, intercept, r_value, _, _ = linregress(amplitudes, peak_velocities)
r_squared = r_value ** 2

feature_vector = np.array([
    slope,
    intercept,
    r_squared,
    np.mean(amplitudes),
    np.std(amplitudes),
    np.mean(peak_velocities),
    np.std(peak_velocities)
])

print("\n=== Feature 2: Main Sequence ===")
print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, R²: {r_squared:.4f}")
print(f"Feature vector length: {len(feature_vector)}")
np.save("feature2_main_sequence.npy", feature_vector)
