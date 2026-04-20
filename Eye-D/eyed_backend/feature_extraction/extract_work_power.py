# feature_extraction/extract_work_power.py
# Feature 1: Kinetic Magnitude – Total Work done during a squeeze.
# Work = ∫ Power dt, where Power = acceleration * velocity (mass = 1)
# Normalised by interocular distance to be scale‑invariant.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# Landmark indices (MediaPipe FaceMesh)
UPPER_LID = 159      # Upper eyelid centre
LOWER_LID = 145      # Lower eyelid centre
LEFT_CANTHUS = 33    # Left eye outer corner
RIGHT_CANTHUS = 133  # Left eye inner corner (we use left eye only)

# Smoothing parameters
SG_WINDOW = 15
SG_ORDER = 3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def get_interocular_distance(landmarks, frame_shape):
    """Return the distance between the two eye corners (normalisation reference)."""
    h, w = frame_shape[:2]
    left = landmarks[LEFT_CANTHUS]
    right = landmarks[RIGHT_CANTHUS]
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    return np.linalg.norm(left_pt - right_pt)

def extract_total_work(video_path):
    """
    Process a single squeeze video and return the total work (normalised).
    Returns a float, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    # Store normalised eyelid distances over time
    norm_distances = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue

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
        return None

    # Smooth the distance signal
    dist_smooth = savgol_filter(norm_distances, SG_WINDOW, SG_ORDER)

    # Velocity = derivative of distance
    velocity = np.gradient(dist_smooth) / dt

    # Acceleration = derivative of velocity
    acceleration = np.gradient(velocity) / dt

    # Power = acceleration * velocity (mass = 1, normalised)
    power = acceleration * velocity

    # Work = ∫ Power dt (use absolute power to capture effort)
    total_work = np.trapz(np.abs(power), dx=dt)

    return float(total_work)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_work_power.py <video_path>")
        sys.exit(1)
    work = extract_total_work(sys.argv[1])
    if work is None:
        print("Extraction failed.")
    else:
        print(f"Total Work: {work:.6f}")
