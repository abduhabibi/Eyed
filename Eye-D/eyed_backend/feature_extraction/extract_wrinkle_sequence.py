# extract_wrinkle_sequence.py
# Extracts the progressive sequential deformation (wrinkle lines) during an eye squeeze.
# Tracks appearance order, timing, positions, lengths, and intensities of each wrinkle.
# Outputs: wrinkle sequence feature vector for AI authentication.

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cdist
import os

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

INPUT_VIDEO_PATH = "squeeze_video.avi"   # <-- CHANGE TO YOUR VIDEO FILE
VIDEO_FPS = 240.0                        # expected frame rate

# MediaPipe landmark indices for eye region (with refine_landmarks=True)
# We'll use the eye region bounding box to isolate area for wrinkle detection
LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]  # landmarks around left eye
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466] # landmarks around right eye

# Canny edge detection parameters (tune these based on your video quality)
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 100

# Minimum contour length to be considered a wrinkle (pixels)
MIN_WRINKLE_LENGTH = 10

# Maximum distance between contours in consecutive frames to consider them the same wrinkle
MAX_TRACKING_DISTANCE = 30

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
# 3. HELPER FUNCTIONS
# -------------------------------

def get_eye_region_mask(landmarks, frame_shape, eye_indices):
    """
    Creates a binary mask for the eye region based on facial landmarks.
    Returns the mask and the bounding box coordinates.
    """
    h, w = frame_shape[:2]
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    
    # Get bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    
    return mask, (x_min, y_min, x_max, y_max)

def detect_wrinkles(image, mask, min_length=MIN_WRINKLE_LENGTH):
    """
    Detects wrinkles in the eye region using Canny edge detection.
    Returns a list of contours (wrinkles) and their properties.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    # Apply mask to isolate eye region
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by length (arc length)
    wrinkles = []
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length >= min_length:
            # Get contour properties
            x, y, w, h = cv2.boundingRect(cnt)
            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            wrinkles.append({
                'contour': cnt,
                'length': length,
                'centroid': (cx, cy),
                'bbox': (x, y, w, h)
            })
    
    return wrinkles

def match_wrinkles(prev_wrinkles, curr_wrinkles, max_dist=MAX_TRACKING_DISTANCE):
    """
    Matches wrinkles between consecutive frames using Hungarian algorithm.
    Returns a list of matched pairs (prev_index, curr_index) and unmatched indices.
    """
    if not prev_wrinkles or not curr_wrinkles:
        return [], list(range(len(prev_wrinkles))), list(range(len(curr_wrinkles)))
    
    # Compute distance matrix between centroids
    prev_centroids = np.array([w['centroid'] for w in prev_wrinkles])
    curr_centroids = np.array([w['centroid'] for w in curr_wrinkles])
    dist_matrix = cdist(prev_centroids, curr_centroids)
    
    # Greedy matching (simple approach)
    matches = []
    used_prev = set()
    used_curr = set()
    
    # Sort by smallest distances first
    for i in range(len(prev_centroids)):
        for j in range(len(curr_centroids)):
            if i not in used_prev and j not in used_curr and dist_matrix[i][j] < max_dist:
                matches.append((i, j))
                used_prev.add(i)
                used_curr.add(j)
                break
    
    unmatched_prev = [i for i in range(len(prev_wrinkles)) if i not in used_prev]
    unmatched_curr = [j for j in range(len(curr_wrinkles)) if j not in used_curr]
    
    return matches, unmatched_prev, unmatched_curr

# -------------------------------
# 4. OPEN VIDEO AND PROCESS
# -------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open {INPUT_VIDEO_PATH}")
    exit()

actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:
    actual_fps = VIDEO_FPS
print(f"Using FPS = {actual_fps}")

# Data storage
timestamps = []                     # time for each frame
wrinkle_events = []                 # list of events (appearance, disappearance)
wrinkle_tracks = {}                 # track each wrinkle's lifespan and properties

frame_idx = 0
prev_wrinkles = None
next_wrinkle_id = 0

print("Processing video frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        timestamp = frame_idx / actual_fps
        timestamps.append(timestamp)
        
        # Get eye region (use right eye for consistency with other features)
        mask, bbox = get_eye_region_mask(landmarks, frame.shape, RIGHT_EYE_INDICES)
        
        # Detect wrinkles in eye region
        current_wrinkles = detect_wrinkles(frame, mask)
        
        # Track wrinkles across frames
        if prev_wrinkles is not None:
            matches, unmatched_prev, unmatched_curr = match_wrinkles(prev_wrinkles, current_wrinkles)
            
            # Update existing wrinkles
            for prev_idx, curr_idx in matches:
                wrinkle_id = prev_wrinkles[prev_idx].get('id')
                if wrinkle_id is not None:
                    # Update wrinkle properties
                    wrinkle_tracks[wrinkle_id]['lengths'].append(current_wrinkles[curr_idx]['length'])
                    wrinkle_tracks[wrinkle_id]['positions'].append(current_wrinkles[curr_idx]['centroid'])
                    wrinkle_tracks[wrinkle_id]['intensities'].append(
                        current_wrinkles[curr_idx]['length'] / prev_wrinkles[prev_idx]['length']
                    )
                    wrinkle_tracks[wrinkle_id]['end_frame'] = frame_idx
                    wrinkle_tracks[wrinkle_id]['end_time'] = timestamp
                    # Assign ID to current wrinkle for next frame
                    current_wrinkles[curr_idx]['id'] = wrinkle_id
            
            # Handle disappeared wrinkles (unmatched in previous)
            for prev_idx in unmatched_prev:
                wrinkle_id = prev_wrinkles[prev_idx].get('id')
                if wrinkle_id is not None:
                    # Mark as disappeared
                    wrinkle_tracks[wrinkle_id]['disappeared_frame'] = frame_idx
                    wrinkle_tracks[wrinkle_id]['disappeared_time'] = timestamp
            
            # Handle new wrinkles (unmatched in current)
            for curr_idx in unmatched_curr:
                new_id = next_wrinkle_id
                next_wrinkle_id += 1
                current_wrinkles[curr_idx]['id'] = new_id
                # Initialize track
                wrinkle_tracks[new_id] = {
                    'appearance_frame': frame_idx,
                    'appearance_time': timestamp,
                    'disappeared_frame': None,
                    'disappeared_time': None,
                    'lengths': [current_wrinkles[curr_idx]['length']],
                    'positions': [current_wrinkles[curr_idx]['centroid']],
                    'intensities': [1.0],  # initial intensity
                    'max_length': current_wrinkles[curr_idx]['length'],
                    'start_bbox': current_wrinkles[curr_idx]['bbox']
                }
                # Record appearance event
                wrinkle_events.append({
                    'type': 'appearance',
                    'wrinkle_id': new_id,
                    'time': timestamp,
                    'frame': frame_idx,
                    'length': current_wrinkles[curr_idx]['length'],
                    'position': current_wrinkles[curr_idx]['centroid']
                })
        else:
            # First frame: initialize all wrinkles as new
            for w in current_wrinkles:
                new_id = next_wrinkle_id
                next_wrinkle_id += 1
                w['id'] = new_id
                wrinkle_tracks[new_id] = {
                    'appearance_frame': frame_idx,
                    'appearance_time': timestamp,
                    'disappeared_frame': None,
                    'disappeared_time': None,
                    'lengths': [w['length']],
                    'positions': [w['centroid']],
                    'intensities': [1.0],
                    'max_length': w['length'],
                    'start_bbox': w['bbox']
                }
                wrinkle_events.append({
                    'type': 'appearance',
                    'wrinkle_id': new_id,
                    'time': timestamp,
                    'frame': frame_idx,
                    'length': w['length'],
                    'position': w['centroid']
                })
        
        # Update for next iteration
        prev_wrinkles = current_wrinkles
    
    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  Processed {frame_idx} frames")

cap.release()

# -------------------------------
# 5. POST-PROCESSING: COMPUTE WRINKLE SEQUENCE FEATURES
# -------------------------------

# Filter out wrinkles that appeared and disappeared during the squeeze
# (ignore those that existed before the squeeze started)
active_wrinkles = {wid: track for wid, track in wrinkle_tracks.items() 
                   if track['disappeared_time'] is not None or len(track['lengths']) > 5}

# Sort wrinkles by appearance time
sorted_wrinkles = sorted(active_wrinkles.items(), key=lambda x: x[1]['appearance_time'])

# Extract sequence features
wrinkle_order = []          # order of appearance (wrinkle IDs)
wrinkle_delays = []         # time delays between consecutive appearances
wrinkle_positions = []      # centroid positions at appearance
wrinkle_lengths = []        # initial lengths
wrinkle_intensities = []    # intensity changes over time
wrinkle_durations = []      # how long each wrinkle lasted

prev_time = None
for wid, track in sorted_wrinkles:
    wrinkle_order.append(wid)
    wrinkle_positions.append(track['positions'][0] if track['positions'] else (0, 0))
    wrinkle_lengths.append(track['lengths'][0] if track['lengths'] else 0)
    wrinkle_intensities.append(np.mean(track['intensities']) if track['intensities'] else 0)
    
    # Duration
    if track['disappeared_time'] is not None:
        duration = track['disappeared_time'] - track['appearance_time']
    else:
        duration = timestamps[-1] - track['appearance_time'] if timestamps else 0
    wrinkle_durations.append(duration)
    
    # Delay from previous wrinkle
    if prev_time is not None:
        wrinkle_delays.append(track['appearance_time'] - prev_time)
    else:
        wrinkle_delays.append(0)
    prev_time = track['appearance_time']

# Normalize positions relative to eye bounding box
normalized_positions = []
if active_wrinkles and timestamps:
    # Get eye region from first valid frame
    # (simplified: use normalized coordinates directly)
    for pos in wrinkle_positions:
        normalized_positions.extend([pos[0], pos[1]])  # flatten

# Create feature vector for AI
# Components:
# - wrinkle_count: total number of wrinkles that appeared
# - wrinkle_order: sequence of wrinkle IDs (as categorical, but we'll use order indices)
# - wrinkle_delays: time between appearances (normalized to total duration)
# - wrinkle_lengths: initial lengths of each wrinkle
# - wrinkle_intensities: average intensity for each wrinkle
# - wrinkle_durations: how long each wrinkle persisted

total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
normalized_delays = [d / total_duration for d in wrinkle_delays] if total_duration > 0 else wrinkle_delays

# Build feature vector
feature_vector = np.array([
    len(wrinkle_order),                     # number of wrinkles
    np.mean(normalized_delays) if normalized_delays else 0,  # average delay
    np.std(normalized_delays) if len(normalized_delays) > 1 else 0,  # delay variation
    np.mean(wrinkle_lengths) if wrinkle_lengths else 0,   # average initial length
    np.std(wrinkle_lengths) if len(wrinkle_lengths) > 1 else 0,  # length variation
    np.mean(wrinkle_intensities) if wrinkle_intensities else 0,  # average intensity
    np.std(wrinkle_intensities) if len(wrinkle_intensities) > 1 else 0,  # intensity variation
    np.mean(wrinkle_durations) if wrinkle_durations else 0,   # average duration
    np.std(wrinkle_durations) if len(wrinkle_durations) > 1 else 0,  # duration variation
] + normalized_positions[:20])  # first 10 positions (20 coordinates)

# -------------------------------
# 6. OUTPUT RESULTS
# -------------------------------

print("\n=== Feature 5: Progressive Sequential Deformation (Wrinkle Sequence) ===")
print(f"Total wrinkles detected: {len(wrinkle_order)}")
print(f"Wrinkle appearance order: {wrinkle_order}")
print(f"Time delays between appearances (s): {wrinkle_delays}")
print(f"Initial wrinkle lengths (pixels): {wrinkle_lengths}")
print(f"Wrinkle intensities: {wrinkle_intensities}")
print(f"Wrinkle durations (s): {wrinkle_durations}")
print(f"\nFeature vector length: {len(feature_vector)}")
print(f"First 10 values: {feature_vector[:10]}")

# Save feature vector
np.save("feature5_wrinkle_sequence.npy", feature_vector)

# Also save full wrinkle tracks for detailed analysis
np.save("wrinkle_tracks.npy", wrinkle_tracks)

# -------------------------------
# 7. OPTIONAL: VISUALIZE WRINKLE DETECTION
# -------------------------------
try:
    import matplotlib.pyplot as plt
    
    # Plot wrinkle appearance timeline
    if wrinkle_order:
        plt.figure(figsize=(10, 6))
        for i, (wid, track) in enumerate(sorted_wrinkles):
            app_time = track['appearance_time']
            if track['disappeared_time'] is not None:
                dis_time = track['disappeared_time']
                plt.hlines(y=i, xmin=app_time, xmax=dis_time, linewidth=3, label=f'Wrinkle {wid}')
            else:
                plt.hlines(y=i, xmin=app_time, xmax=timestamps[-1], linewidth=3, label=f'Wrinkle {wid}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Wrinkle ID (order of appearance)')
        plt.title('Wrinkle Appearance and Persistence Timeline')
        plt.grid(True)
        plt.show()
except ImportError:
    print("Matplotlib not available – skipping plot.")
