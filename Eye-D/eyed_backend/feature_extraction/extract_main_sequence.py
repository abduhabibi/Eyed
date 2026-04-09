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
import argparse

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
# The backend uses a fixed input video path (`squeeze_video.avi`) and calls this
# script once per uploaded video. For the original research workflow you can
# still point VIDEO_FOLDER to a directory of squeeze clips, but for the app to
# run out-of-the-box we fall back to the fixed input if the folder is missing or empty.
VIDEO_FOLDER = "squeeze_videos_person_A/"   # optional research dataset folder
DEFAULT_INPUT_VIDEO_PATH = "squeeze_video.avi"
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
# 2. PROCESS VIDEOS (folder if available, else fixed input)
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO_PATH, help="Path to input video (backend mode)")
parser.add_argument("--video-folder", default=VIDEO_FOLDER, help="Folder of squeeze clips (research mode)")
parser.add_argument("--outdir", default=".", help="Directory to write .npy outputs")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

video_paths = []
if os.path.isdir(args.video_folder):
    video_paths = glob.glob(os.path.join(args.video_folder, "*.[amv][pm4]*"))  # .avi, .mp4, .mov

amplitudes, peak_velocities = [], []
for path in video_paths:
    amp, vel = extract_amplitude_peak_velocity(path)
    if amp is not None and vel is not None:
        amplitudes.append(amp)
        peak_velocities.append(vel)

# If we don't have enough clips for a regression, fall back to a single-clip estimate.
if len(amplitudes) >= 3:
    slope, intercept, r_value, _, _ = linregress(amplitudes, peak_velocities)
    r_squared = r_value ** 2
else:
    amp, vel = extract_amplitude_peak_velocity(args.input)
    if amp is None or vel is None:
        print("Need at least one valid squeeze video.")
        exit()
    amplitudes = [amp]
    peak_velocities = [vel]
    slope, intercept, r_squared = 0.0, 0.0, 0.0

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
np.save(os.path.join(args.outdir, "feature2_main_sequence.npy"), feature_vector)
