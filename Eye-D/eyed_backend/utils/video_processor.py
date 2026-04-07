# utils/video_processor.py
# Common video handling utilities for the Eye-D extraction pipeline.
# Functions:
#   - load_video_info(path) : returns fps, total frames, duration.
#   - get_frame_generator(path) : yields frames with timestamps.
#   - stabilize_eye_region(frame, landmarks) : warps eye region to a canonical size.
#   - extract_eye_roi(frame, landmarks, eye_side='right') : crops the eye area.

import cv2
import numpy as np
import mediapipe as mp

# -------------------------------
# 1. BASIC VIDEO METADATA
# -------------------------------
def load_video_info(video_path: str) -> dict:
    """
    Opens a video file and returns its properties.
    Returns dictionary: {'fps': float, 'frame_count': int, 'duration': float}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {'fps': fps, 'frame_count': frame_count, 'duration': duration}

def get_frame_generator(video_path: str):
    """
    Generator that yields (frame, timestamp_seconds) for each video frame.
    Timestamp is based on frame index and FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        yield frame, timestamp
        frame_idx += 1
    cap.release()

# -------------------------------
# 2. EYE REGION EXTRACTION AND STABILIZATION
# -------------------------------
def extract_eye_roi(frame: np.ndarray, landmarks, eye_side: str = 'right', padding: int = 20):
    """
    Given a frame and MediaPipe face landmarks, returns the cropped eye region
    (as a sub-image) and its bounding box (x, y, w, h) in the original frame.
    eye_side: 'right' or 'left'.
    """
    h, w = frame.shape[:2]
    # Landmark indices for right eye (with refine_landmarks=True)
    if eye_side == 'right':
        # Indices that outline the right eye (iris, corners, lids)
        eye_indices = [33, 133, 157, 158, 159, 160, 161, 173]
    else:  # left eye
        eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max - x_min, y_max - y_min)

def stabilize_eye_region(frame: np.ndarray, landmarks, target_size=(128, 64)):
    """
    Uses eye corner landmarks to warp the eye region to a canonical size and orientation.
    This compensates for small head rotations, making features more stable.
    Returns the stabilized eye image (grayscale).
    """
    h, w = frame.shape[:2]
    # For right eye: inner canthus (133) and outer canthus (362)
    inner = (int(landmarks[133].x * w), int(landmarks[133].y * h))
    outer = (int(landmarks[362].x * w), int(landmarks[362].y * h))
    # Desired positions after warp: inner -> (0.2*width, 0.5*height), outer -> (0.8*width, 0.5*height)
    src_pts = np.float32([inner, outer])
    dst_pts = np.float32([[0.2 * target_size[0], target_size[1] / 2],
                          [0.8 * target_size[0], target_size[1] / 2]])
    # Estimate affine transform (rotation, scaling, translation)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    # Warp the whole frame (or just the eye region) – here we warp the frame
    warped = cv2.warpAffine(frame, M, target_size, flags=cv2.INTER_LINEAR)
    # Convert to grayscale for feature extraction
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray

# -------------------------------
# 3. OPTICAL FLOW FOR HEAD MOVEMENT COMPENSATION
# -------------------------------
def compute_face_shift(prev_landmarks, curr_landmarks, frame_shape):
    """
    Computes the translation of the face centroid between two frames.
    Returns (dx, dy) in pixels. Used to cancel out head movement.
    """
    h, w = frame_shape[:2]
    # Use nose tip (landmark 1) or face center
    nose_idx = 1
    prev_x = prev_landmarks[nose_idx].x * w
    prev_y = prev_landmarks[nose_idx].y * h
    curr_x = curr_landmarks[nose_idx].x * w
    curr_y = curr_landmarks[nose_idx].y * h
    return curr_x - prev_x, curr_y - prev_y
