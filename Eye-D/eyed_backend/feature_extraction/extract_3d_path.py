# feature2_medial_canthus_3d.py
# Extracts 3D trajectory of medial canthus (inner eye corner) from video using MediaPipe.
# Assumes video recorded at 240 FPS.

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

INPUT_VIDEO_PATH = "squeeze_video.avi"   # <-- CHANGE TO YOUR VIDEO FILE
VIDEO_FPS = 240.0                        # expected frame rate

# MediaPipe landmark index for medial canthus (inner corner of right eye)
# With refine_landmarks=True, landmark 133 = right eye inner corner
MEDIAL_CANTHUS_ID = 133

# Savitzky-Golay filter for smoothing 3D coordinates (optional but recommended)
SG_WINDOW = 15       # must be odd and <= number of frames
SG_ORDER = 3

# -------------------------------
# 2. INITIALIZE MEDIAPIPE FACE MESH
# -------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,      # gives precise eye landmarks including inner canthus
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
# 4. DATA LISTS
# -------------------------------
timestamps = []      # time in seconds
trajectory_3d = []   # list of (x, y, z) normalized coordinates (0..1)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Get medial canthus landmark (normalized x, y, z)
        point = landmarks[MEDIAL_CANTHUS_ID]
        trajectory_3d.append([point.x, point.y, point.z])
        timestamps.append(frame_idx / actual_fps)

    frame_idx += 1

cap.release()

# Convert to numpy arrays
traj = np.array(trajectory_3d)   # shape (n_frames, 3)
timestamps = np.array(timestamps)

if len(traj) < SG_WINDOW:
    print("Not enough frames for smoothing.")
    exit()

# -------------------------------
# 5. SMOOTH EACH COORDINATE SEPARATELY
# -------------------------------
# Apply Savitzky-Golay filter to x, y, z independently to reduce noise
traj_smooth = np.zeros_like(traj)
for i in range(3):   # for x, y, z
    traj_smooth[:, i] = savgol_filter(traj[:, i], window_length=SG_WINDOW, polyorder=SG_ORDER)

# -------------------------------
# 6. COMPUTE 3D PATH LENGTH (total distance traveled by medial canthus)
# -------------------------------
# Path length = sum of Euclidean distances between consecutive smoothed points
path_length = 0.0
for i in range(1, len(traj_smooth)):
    dist = euclidean(traj_smooth[i], traj_smooth[i-1])
    path_length += dist

# -------------------------------
# 7. COMPUTE CURVATURE (change in direction per unit time)
# -------------------------------
# Approximate curvature by angular change between consecutive velocity vectors.
# Velocity vectors = difference between consecutive positions.
velocities = np.diff(traj_smooth, axis=0)   # shape (n_frames-1, 3)
# Compute angle between each consecutive velocity vector (in radians)
angles = []
for i in range(1, len(velocities)):
    v1 = velocities[i-1]
    v2 = velocities[i]
    # Avoid division by zero if velocity magnitude is zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        angles.append(angle)
    else:
        angles.append(0.0)
angles = np.array(angles)
# Average curvature (radians per frame) – can also integrate over time
mean_curvature = np.mean(angles) if len(angles) > 0 else 0.0

# -------------------------------
# 8. MAXIMUM DISPLACEMENT FROM STARTING POSITION
# -------------------------------
start_point = traj_smooth[0]
displacements = np.linalg.norm(traj_smooth - start_point, axis=1)
max_displacement = np.max(displacements)

# -------------------------------
# 9. 3D TRAJECTORY AS A SEQUENCE (normalized to fixed length for AI)
# -------------------------------
# For AI input, we need a fixed-length representation.
# Resample the smoothed trajectory to, say, 100 equally spaced time points.
# We'll use linear interpolation.
num_target_points = 100
target_times = np.linspace(timestamps[0], timestamps[-1], num_target_points)
traj_resampled = np.zeros((num_target_points, 3))
for i in range(3):
    traj_resampled[:, i] = np.interp(target_times, timestamps, traj_smooth[:, i])

# Flatten the resampled trajectory into a 1D feature vector
# Shape: (100*3) = 300 numbers
trajectory_feature = traj_resampled.flatten()

# -------------------------------
# 10. OUTPUT FEATURE VECTOR FOR THIS SQUEEZE
# -------------------------------
print("\n=== Feature 2: Medial Canthus 3D Path ===")
print(f"Total path length (normalized units): {path_length:.4f}")
print(f"Mean curvature (radians per frame): {mean_curvature:.4f}")
print(f"Maximum displacement from start: {max_displacement:.4f}")
print(f"Trajectory feature vector length: {len(trajectory_feature)} numbers")
print(f"First 10 values: {trajectory_feature[:10]}")

# Optionally save feature vector to file for later use with AI
np.save("feature2_medial_canthus.npy", trajectory_feature)

# -------------------------------
# 11. OPTIONAL: VISUALIZE 3D TRAJECTORY
# -------------------------------
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_smooth[:, 0], traj_smooth[:, 1], traj_smooth[:, 2], 'b-', linewidth=1)
    ax.scatter(traj_smooth[0,0], traj_smooth[0,1], traj_smooth[0,2], c='g', marker='o', label='Start')
    ax.scatter(traj_smooth[-1,0], traj_smooth[-1,1], traj_smooth[-1,2], c='r', marker='x', label='End')
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (normalized depth)')
    ax.set_title('3D Trajectory of Medial Canthus During Eye Squeeze')
    ax.legend()
    plt.show()
except ImportError:
    print("Matplotlib not available for 3D plot.")
