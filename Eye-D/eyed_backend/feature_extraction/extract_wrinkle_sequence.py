# extract_wrinkle_sequence.py
# Feature 4: Progressive sequential deformation (wrinkles).
# Normalizes positions and lengths by interocular distance.
# Uses histogram equalization to improve wrinkle visibility in poor lighting.

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cdist
import argparse
import os

DEFAULT_INPUT_VIDEO_PATH = "squeeze_video.avi"
LEFT_CANTHUS = 133
RIGHT_CANTHUS = 362
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]

CANNY_LOW = 30
CANNY_HIGH = 100
MIN_WRINKLE_LEN = 10   # pixels, but will be normalized
MAX_TRACKING_DIST = 30

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_interocular_distance(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left = landmarks[LEFT_CANTHUS]
    right = landmarks[RIGHT_CANTHUS]
    left_pt = np.array([left.x * w, left.y * h])
    right_pt = np.array([right.x * w, right.y * h])
    return np.linalg.norm(left_pt - right_pt)

def get_eye_roi(landmarks, frame_shape):
    h, w = frame_shape[:2]
    pts = []
    for idx in RIGHT_EYE_LANDMARKS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])
    pts = np.array(pts)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    pad = 20
    x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
    x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)
    return (x_min, y_min, x_max, y_max)

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO_PATH, help="Path to input video")
parser.add_argument("--outdir", default=".", help="Directory to write .npy outputs")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

cap = cv2.VideoCapture(args.input)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

frame_idx = 0
wrinkle_tracks = {}   # id -> {appearance_time, positions_norm, lengths_norm, ...}
next_id = 0
prev_wrinkles = []

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
        # Get eye ROI
        roi = get_eye_roi(landmarks, frame.shape)
        x1, y1, x2, y2 = roi
        eye_img = frame[y1:y2, x1:x2]
        # Improve lighting
        eye_img = cv2.equalizeHist(cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY))
        # Detect wrinkles
        edges = cv2.Canny(eye_img, CANNY_LOW, CANNY_HIGH)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_wrinkles = []
        for cnt in contours:
            length_px = cv2.arcLength(cnt, closed=False)
            if length_px >= MIN_WRINKLE_LEN:
                length_norm = length_px / ref_dist
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                else:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                # Normalize position by ref_dist
                pos_norm = (cx / ref_dist, cy / ref_dist)
                current_wrinkles.append({'length_norm': length_norm, 'pos_norm': pos_norm})
        # Simple tracking (match by closest centroid)
        if prev_wrinkles:
            # Greedy matching
            used_prev = set()
            for i, curr in enumerate(current_wrinkles):
                best_dist = float('inf')
                best_j = -1
                for j, prev in enumerate(prev_wrinkles):
                    if j in used_prev:
                        continue
                    dist = np.linalg.norm(np.array(curr['pos_norm']) - np.array(prev['pos_norm']))
                    if dist < MAX_TRACKING_DIST and dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j != -1:
                    used_prev.add(best_j)
                    if 'id' in prev_wrinkles[best_j]:
                        curr['id'] = prev_wrinkles[best_j]['id']
            # New wrinkles
            for curr in current_wrinkles:
                if 'id' not in curr:
                    curr['id'] = next_id
                    next_id += 1
                    wrinkle_tracks[curr['id']] = {
                        'appearance_time': frame_idx / fps,
                        'positions': [],
                        'lengths': [],
                        'max_length': 0
                    }
                wrinkle_tracks[curr['id']]['positions'].append(curr['pos_norm'])
                wrinkle_tracks[curr['id']]['lengths'].append(curr['length_norm'])
                if curr['length_norm'] > wrinkle_tracks[curr['id']]['max_length']:
                    wrinkle_tracks[curr['id']]['max_length'] = curr['length_norm']
        prev_wrinkles = current_wrinkles
    frame_idx += 1
cap.release()

# Build feature vector: number of wrinkles, avg length, avg delay, etc.
if not wrinkle_tracks:
    print("No wrinkles detected")
    exit()

wrinkle_order = sorted(wrinkle_tracks.keys(), key=lambda x: wrinkle_tracks[x]['appearance_time'])
delays = []
for i in range(1, len(wrinkle_order)):
    t_prev = wrinkle_tracks[wrinkle_order[i-1]]['appearance_time']
    t_curr = wrinkle_tracks[wrinkle_order[i]]['appearance_time']
    delays.append(t_curr - t_prev)
avg_delay = np.mean(delays) if delays else 0
avg_length = np.mean([wrinkle_tracks[w]['max_length'] for w in wrinkle_order])
num_wrinkles = len(wrinkle_order)

feature_vector = np.array([num_wrinkles, avg_delay, avg_length])
print("\n=== Feature 4: Wrinkle Sequence ===")
print(f"Number of wrinkles: {num_wrinkles}")
print(f"Avg delay between appearances (s): {avg_delay:.4f}")
print(f"Avg max wrinkle length (normalized): {avg_length:.4f}")
np.save(os.path.join(args.outdir, "feature4_wrinkle.npy"), feature_vector)
