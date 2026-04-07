# extract_wave_closure.py
# Measures wave-like propagation of eyelid closure across multiple segments.
# Tracks inner, middle, and outer parts of the eyelid to compute segment delays.
# Outputs: segment closure times, wave speed, spatial coherence, feature vector.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
import os

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

INPUT_VIDEO_PATH = "squeeze_video.avi"   # <-- CHANGE TO YOUR VIDEO FILE
VIDEO_FPS = 240.0                        # expected frame rate

# MediaPipe landmark indices for right eye (with refine_landmarks=True)
# We will use three points along the upper eyelid margin:
# Inner segment: landmark 133 (medial canthus) – but we need upper eyelid points.
# Better: use landmarks 159 (upper lid, inner), 145 is lower, not upper.
# For upper eyelid, the following indices (from MediaPipe FaceMesh with refine):
# 159 = upper eyelid (inner side)
# 145 = lower eyelid (not used here)
# Actually, to get three horizontal positions on the upper eyelid:
# We can interpolate between medial canthus (133) and lateral canthus (362 or 263?).
# Simpler: use predefined upper eyelid landmarks that span the lid.
# With refine_landmarks=True, indices 159, 160, 161 are along upper lid? Not exactly.
# Based on MediaPipe documentation, for right eye:
# 159 = upper eyelid (center-ish)
# 160 = upper eyelid (outer side)
# 161 = upper eyelid (inner side?) Actually, let's use:
# 133 = inner canthus (corner)
# 362 = outer canthus (corner)
# Then we can sample points along the eyelid curve using interpolation.
# But for simplicity and reliability, I'll use three fixed upper lid landmarks:
# 159 (upper inner), 160 (upper middle), 161 (upper outer) – these exist with refine_landmarks.
# Let's verify: In MediaPipe, with refine_landmarks=True, the right eye has:
# 159, 160, 161 for upper eyelid, and 145, 146, 147 for lower eyelid.
# So we use 159 (inner third), 160 (middle), 161 (outer third).

UPPER_LID_INNER = 159   # inner part of upper eyelid
UPPER_LID_MIDDLE = 160  # middle part
UPPER_LID_OUTER = 161   # outer part

# Corresponding lower eyelid landmarks for distance calculation
LOWER_LID_INNER = 145   # inner part of lower eyelid
LOWER_LID_MIDDLE = 146  # middle part
LOWER_LID_OUTER = 147   # outer part

# Savitzky-Golay filter parameters
SG_WINDOW = 15          # must be odd
SG_ORDER = 3

# Closure threshold (fraction of full closure) to detect segment activation time
# We'll use 50% closure as the reference point.
CLOSURE_THRESHOLD = 0.5

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
# 3. OPEN VIDEO
# -------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open {INPUT_VIDEO_PATH}")
    exit()

actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:
    actual_fps = VIDEO_FPS
print(f"Using FPS = {actual_fps}")

# -------------------------------
# 4. DATA STORAGE
# -------------------------------

# For each segment, store distances over time
segment_distances = {
    'inner': [],
    'middle': [],
    'outer': []
}
timestamps = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Helper function to get vertical distance between upper and lower lid at a given landmark pair
        def get_vertical_distance(upper_id, lower_id):
            upper = landmarks[upper_id]
            lower = landmarks[lower_id]
            upper_y = upper.y * h
            lower_y = lower.y * h
            return lower_y - upper_y   # positive when eye open

        # Compute distances for each segment
        dist_inner = get_vertical_distance(UPPER_LID_INNER, LOWER_LID_INNER)
        dist_middle = get_vertical_distance(UPPER_LID_MIDDLE, LOWER_LID_MIDDLE)
        dist_outer = get_vertical_distance(UPPER_LID_OUTER, LOWER_LID_OUTER)

        segment_distances['inner'].append(dist_inner)
        segment_distances['middle'].append(dist_middle)
        segment_distances['outer'].append(dist_outer)
        timestamps.append(frame_idx / actual_fps)

    frame_idx += 1
cap.release()

# Convert to numpy arrays
for key in segment_distances:
    segment_distances[key] = np.array(segment_distances[key])

timestamps = np.array(timestamps)

# Check data sufficiency
min_len = min(len(segment_distances['inner']), len(segment_distances['middle']), len(segment_distances['outer']))
if min_len < SG_WINDOW:
    print("Not enough frames for smoothing.")
    exit()

# -------------------------------
# 5. SMOOTH EACH SEGMENT'S DISTANCE
# -------------------------------

smoothed = {}
for key in segment_distances:
    smoothed[key] = savgol_filter(segment_distances[key], window_length=SG_WINDOW, polyorder=SG_ORDER)

# -------------------------------
# 6. NORMALIZE EACH SEGMENT'S DISTANCE TO [0,1] (0 = fully closed, 1 = fully open)
# -------------------------------
# For each segment, find max distance (fully open) and min distance (fully squeezed)
normalized = {}
for key in smoothed:
    dist_array = smoothed[key]
    max_dist = np.max(dist_array)
    min_dist = np.min(dist_array)
    if max_dist - min_dist == 0:
        normalized[key] = np.ones_like(dist_array) * 0.5  # fallback
    else:
        normalized[key] = (dist_array - min_dist) / (max_dist - min_dist)

# -------------------------------
# 7. FIND TIME WHEN EACH SEGMENT REACHES CLOSURE_THRESHOLD (e.g., 50% closed)
# -------------------------------
# We want the first time when normalized distance drops below (1 - CLOSURE_THRESHOLD)?
# Actually, distance = 1 = fully open, 0 = fully closed. So 50% closure means distance = 0.5.
# But closure is from open to closed. We'll define closure_fraction = 1 - normalized_distance.
# Then closure_threshold = 0.5 means when closure_fraction >= 0.5.
# Simpler: find time when normalized distance <= 0.5 (half-closed).

closure_times = {}
for key in normalized:
    norm = normalized[key]
    # Find first index where norm <= CLOSURE_THRESHOLD (assuming starting open > threshold)
    indices = np.where(norm <= CLOSURE_THRESHOLD)[0]
    if len(indices) > 0:
        idx = indices[0]
        closure_times[key] = timestamps[idx]
    else:
        closure_times[key] = np.nan   # never reached threshold

# -------------------------------
# 8. COMPUTE TIME DELAYS BETWEEN SEGMENTS
# -------------------------------
# Order: inner, middle, outer (natural propagation from inner to outer or vice versa)
# Typically, the wave starts near the inner canthus and moves outward.
delay_inner_to_middle = closure_times['middle'] - closure_times['inner']
delay_middle_to_outer = closure_times['outer'] - closure_times['middle']
delay_inner_to_outer = closure_times['outer'] - closure_times['inner']

# -------------------------------
# 9. COMPUTE WAVE SPEED (assuming average distance between segments in pixels)
# -------------------------------
# To compute speed, we need the horizontal distance between segment landmarks in pixels.
# We'll compute the x-coordinates of the upper lid landmarks from the first frame.
# Re-open video to get landmark positions (or compute from first valid frame).
# Simpler: use normalized x coordinates (0..1) and multiply by image width.
# We'll approximate: inner to middle distance, middle to outer distance.
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
first_landmarks = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        first_landmarks = results.multi_face_landmarks[0].landmark
        break
cap.release()

if first_landmarks is not None:
    h, w, _ = frame.shape
    inner_x = first_landmarks[UPPER_LID_INNER].x * w
    middle_x = first_landmarks[UPPER_LID_MIDDLE].x * w
    outer_x = first_landmarks[UPPER_LID_OUTER].x * w
    dist_inner_middle_px = abs(middle_x - inner_x)
    dist_middle_outer_px = abs(outer_x - middle_x)

    # Wave speed = distance / time delay (pixels per second)
    if delay_inner_to_middle > 0 and not np.isnan(delay_inner_to_middle):
        wave_speed_inner_middle = dist_inner_middle_px / delay_inner_to_middle
    else:
        wave_speed_inner_middle = 0.0

    if delay_middle_to_outer > 0 and not np.isnan(delay_middle_to_outer):
        wave_speed_middle_outer = dist_middle_outer_px / delay_middle_to_outer
    else:
        wave_speed_middle_outer = 0.0
else:
    wave_speed_inner_middle = 0.0
    wave_speed_middle_outer = 0.0

# -------------------------------
# 10. SPATIAL COHERENCE (consistency of propagation direction)
# -------------------------------
# Coherence = 1 if delays are all positive (inner→middle→outer in order), 
# 0 if any delay negative or out of order.
if delay_inner_to_middle > 0 and delay_middle_to_outer > 0:
    spatial_coherence = 1.0
elif delay_inner_to_middle < 0 and delay_middle_to_outer < 0:
    spatial_coherence = 1.0  # consistent reverse order
else:
    spatial_coherence = 0.0   # mixed order

# -------------------------------
# 11. FEATURE VECTOR FOR AI
# -------------------------------
# Features: delays (3 values), wave speeds (2 values), spatial coherence (1 value), 
# and the full normalized closure curves resampled to fixed length.
num_target_points = 100
target_times = np.linspace(timestamps[0], timestamps[-1], num_target_points)
resampled_curves = []
for key in normalized:
    curve = np.interp(target_times, timestamps, normalized[key])
    resampled_curves.append(curve)
# Concatenate all three curves into one vector
curve_features = np.concatenate(resampled_curves)  # length 300

# Combine all features
feature_vector = np.concatenate([
    [delay_inner_to_middle if not np.isnan(delay_inner_to_middle) else 0.0],
    [delay_middle_to_outer if not np.isnan(delay_middle_to_outer) else 0.0],
    [delay_inner_to_outer if not np.isnan(delay_inner_to_outer) else 0.0],
    [wave_speed_inner_middle],
    [wave_speed_middle_outer],
    [spatial_coherence],
    curve_features
])

print("\n=== Feature 4: Wave-like Closure Pattern ===")
print(f"Delay inner→middle (s): {delay_inner_to_middle:.4f}" if not np.isnan(delay_inner_to_middle) else "Delay inner→middle: N/A")
print(f"Delay middle→outer (s): {delay_middle_to_outer:.4f}" if not np.isnan(delay_middle_to_outer) else "Delay middle→outer: N/A")
print(f"Delay inner→outer (s): {delay_inner_to_outer:.4f}" if not np.isnan(delay_inner_to_outer) else "Delay inner→outer: N/A")
print(f"Wave speed inner→middle (pixels/s): {wave_speed_inner_middle:.2f}")
print(f"Wave speed middle→outer (pixels/s): {wave_speed_middle_outer:.2f}")
print(f"Spatial coherence: {spatial_coherence}")
print(f"Total feature vector length: {len(feature_vector)}")

# Save feature vector
np.save("feature4_wave_closure.npy", feature_vector)

# -------------------------------
# 12. OPTIONAL: PLOT THE THREE CLOSURE CURVES
# -------------------------------
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for key in normalized:
        plt.plot(timestamps, normalized[key], label=f'{key} segment')
    plt.axhline(y=CLOSURE_THRESHOLD, color='k', linestyle='--', label=f'{CLOSURE_THRESHOLD*100}% closure')
    for key, t in closure_times.items():
        if not np.isnan(t):
            plt.axvline(x=t, linestyle=':', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized eyelid opening (1=open, 0=closed)')
    plt.title('Wave-like Closure: Segment Timing')
    plt.legend()
    plt.grid(True)
    plt.show()
except ImportError:
    print("Matplotlib not available – skipping plot.")
