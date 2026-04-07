# extract_work_power.py
# Feature 1: Kinetic Magnitude – computes normalized distance, velocity, acceleration, power, work.
# Scale‑invariant using interocular distance normalization.
# Handles head movement via landmark‑based stabilization.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
INPUT_VIDEO_PATH = "squeeze_video.avi"   # CHANGE THIS
EYELID_MASS_KG = 0.0003                  # ~0.3 grams

# MediaPipe landmarks (right eye, refine_landmarks=True)
UPPER_LID = 159
LOWER_LID = 145
LEFT_CANTHUS = 133    # right eye inner corner (medial)
RIGHT_CANTHUS = 362   # left eye inner corner – for interocular distance

# Savitzky‑Golay filter
SG_WINDOW = 15        # must be odd
SG_ORDER = 3

# -------------------------------
# 2. INITIALIZE MEDIAPIPE
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# 3. HELPER: COMPUTE INTEROCULAR DISTANCE (pixels)
# -------------------------------
def get_interocular_distance(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left = landmarks[LEFT_CANTHUS]
    right = landmarks[RIGHT_CANTHUS]
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    return np.linalg.norm(left_pt - right_pt)

# -------------------------------
# 4. PROCESS VIDEO
# -------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open {INPUT_VIDEO_PATH}")
    exit()

# Get actual FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
    print("Warning: Using default FPS = 30")

timestamps = []
norm_distances = []   # normalized eyelid distance (dimensionless)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Get interocular distance (reference for normalization)
        ref_dist = get_interocular_distance(landmarks, frame.shape)
        if ref_dist < 1:
            continue   # skip if face too small or detection bad

        # Upper and lower eyelid y‑coordinates (pixels)
        upper_y = landmarks[UPPER_LID].y * h
        lower_y = landmarks[LOWER_LID].y * h
        dist_px = lower_y - upper_y
        norm_dist = dist_px / ref_dist   # dimensionless

        timestamps.append(frame_idx / fps)
        norm_distances.append(norm_dist)

    frame_idx += 1
cap.release()

if len(norm_distances) < SG_WINDOW:
    print("Not enough frames. Check video or landmarks.")
    exit()

# -------------------------------
# 5. SMOOTH & COMPUTE KINEMATICS
# -------------------------------
dist_smooth = savgol_filter(norm_distances, SG_WINDOW, SG_ORDER)
dt = 1.0 / fps
velocity = np.gradient(dist_smooth) / dt          # dimensionless/s
acceleration = np.gradient(velocity) / dt         # dimensionless/s²
force = EYELID_MASS_KG * acceleration             # kg * (dimensionless/s²) → arbitrary units
power = force * velocity                          # arbitrary units
work = trapezoid(power, timestamps)               # integral over time

# Braking phase (most negative power)
min_power_idx = np.argmin(power)
braking_time = timestamps[min_power_idx]
braking_power = power[min_power_idx]

# Feature vector for AI
# [peak_velocity, peak_acceleration, work, braking_power, braking_time_ratio, duration]
peak_velocity = np.max(np.abs(velocity))
peak_acceleration = np.max(np.abs(acceleration))
duration = timestamps[-1] - timestamps[0]
braking_time_ratio = braking_time / duration if duration > 0 else 0

feature_vector = np.array([
    peak_velocity,
    peak_acceleration,
    work,
    braking_power,
    braking_time_ratio,
    duration
])

print("\n=== Feature 1: Kinetic Magnitude ===")
print(f"Peak velocity (dimensionless/s): {peak_velocity:.4f}")
print(f"Peak acceleration (dimensionless/s²): {peak_acceleration:.4f}")
print(f"Work (arbitrary units): {work:.4f}")
print(f"Braking time ratio: {braking_time_ratio:.4f}")
print(f"Duration (s): {duration:.4f}")
print(f"Feature vector length: {len(feature_vector)}")
np.save("feature1_kinetic.npy", feature_vector)
