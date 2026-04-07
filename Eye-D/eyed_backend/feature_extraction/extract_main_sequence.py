# eyelid_squeeze_analysis.py
# Extracts kinetic features from eyelid squeeze using MediaPipe, NumPy, SciPy.
# Assumes input video recorded at 240 FPS.

import cv2                          # OpenCV for video reading and image display
import mediapipe as mp              # MediaPipe for facial landmark detection
import numpy as np                  # NumPy for numerical operations
from scipy.signal import savgol_filter  # Savitzky-Golay filter for smoothing
from scipy.integrate import trapezoid    # Trapezoidal integration

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

# Estimated mass of an eyelid (kilograms) – a small constant for force calculation
EYELID_MASS_KG = 0.0003   # ~0.3 grams

# Path to your input video (recorded at 240 FPS)
INPUT_VIDEO_PATH = "squeeze_video.avi"   # <-- CHANGE THIS TO YOUR VIDEO FILE

# Desired frame rate (must match video's actual FPS)
VIDEO_FPS = 240.0

# Savitzky-Golay filter parameters
# window_length must be odd and smaller than number of frames
SG_WINDOW = 15           # number of points used for smoothing (adjust based on noise)
SG_ORDER = 3             # polynomial order (typical 2 or 3)

# -------------------------------
# 2. INITIALIZE MEDIAPIPE FACE MESH
# -------------------------------

mp_face_mesh = mp.solutions.face_mesh
# Use static_image_mode=False for video, max_num_faces=1, refine_landmarks=True for better eyelid tracking
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,       # gives additional landmarks around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for right eye (based on MediaPipe FaceMesh with refine_landmarks)
# 159 = upper eyelid, 145 = lower eyelid (right eye)
UPPER_LID_ID = 159
LOWER_LID_ID = 145

# -------------------------------
# 3. OPEN VIDEO FILE
# -------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
    exit()

# Get actual video FPS from the file (fallback to VIDEO_FPS if not available)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:
    actual_fps = VIDEO_FPS
    print(f"Warning: Using user-provided FPS = {actual_fps}")
else:
    print(f"Video FPS = {actual_fps}")

# -------------------------------
# 4. LISTS TO STORE DATA FOR EACH FRAME
# -------------------------------

timestamps = []      # time in seconds for each processed frame
distances = []       # vertical distance between upper and lower eyelid (pixels)

# -------------------------------
# 5. PROCESS VIDEO FRAME BY FRAME
# -------------------------------

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break   # end of video

    # Convert BGR (OpenCV default) to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run MediaPipe face mesh detection
    results = face_mesh.process(rgb_frame)

    # If a face is detected and landmarks exist
    if results.multi_face_landmarks:
        # Get the first face's landmarks (normalized coordinates 0..1)
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract normalized (x,y) of upper and lower eyelid landmarks
        upper = landmarks[UPPER_LID_ID]
        lower = landmarks[LOWER_LID_ID]

        # Convert normalized coordinates to pixel coordinates using frame dimensions
        h, w, _ = frame.shape
        upper_y = upper.y * h   # y-coordinate in pixels (vertical)
        lower_y = lower.y * h

        # Vertical distance in pixels (positive if upper eyelid is above lower)
        distance = lower_y - upper_y   # lower y is greater (downwards) so distance > 0
        distances.append(distance)

        # Compute timestamp for this frame (based on frame index and FPS)
        # Time = frame_number / fps
        # We'll compute using actual_fps from video file
        timestamp = frame_idx / actual_fps
        timestamps.append(timestamp)

    else:
        # No face detected in this frame – we can either skip or append NaN
        # Here we skip the frame entirely (do not append to distances/timestamps)
        # This maintains alignment but reduces sample count.
        # For simplicity, we just ignore frames without a face.
        pass

    frame_idx += 1

    # Optional: display progress every 100 frames
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx} frames")

# Release video capture
cap.release()

# -------------------------------
# 6. CONVERT LISTS TO NUMPY ARRAYS
# -------------------------------

timestamps = np.array(timestamps)
distances = np.array(distances)

if len(distances) < SG_WINDOW:
    print("Error: Not enough valid frames with face detected for smoothing.")
    exit()

print(f"Total frames with valid eyelid detection: {len(distances)}")

# -------------------------------
# 7. SMOOTH DISTANCE DATA (Savitzky-Golay filter)
# -------------------------------
# Smoothing removes high-frequency camera jitter and noise.
smoothed_distances = savgol_filter(distances, window_length=SG_WINDOW, polyorder=SG_ORDER)

# -------------------------------
# 8. COMPUTE VELOCITY (v = dx/dt)
# -------------------------------
# dt between consecutive frames (assuming constant frame rate)
dt = 1.0 / actual_fps   # time step in seconds

# Use numpy gradient to compute derivative (more accurate than simple differences)
# gradient returns derivative with respect to index, then divide by dt
velocity = np.gradient(smoothed_distances) / dt   # units: pixels/second

# -------------------------------
# 9. COMPUTE ACCELERATION (a = dv/dt)
# -------------------------------
acceleration = np.gradient(velocity) / dt          # units: pixels/second²

# -------------------------------
# 10. COMPUTE FORCE (F = m * a)
# -------------------------------
# Convert acceleration from pixels/s² to meters/s²? 
# We need a pixel-to-meter conversion. For now, we'll keep in pixels and note that
# the force and power will be in arbitrary units (pixel-based). 
# If you have a calibration (e.g., known inter-pupillary distance), you can convert.
# Here we assume we work in pixel space – the shape of power curve is what matters.
force = EYELID_MASS_KG * acceleration               # units: kg * (pixels/s²) = "pixel-based Newtons"

# -------------------------------
# 11. COMPUTE POWER (P = Force * Velocity)
# -------------------------------
# Power = force × velocity (dot product, but both are scalars along vertical axis)
power = force * velocity                            # units: (kg * pixels/s²) * (pixels/s) = kg * pixels² / s³

# -------------------------------
# 12. COMPUTE WORK (W = ∫ P dt) using trapezoidal rule
# -------------------------------
# Work = integral of power over time (area under power-time curve)
work = trapezoid(power, timestamps)                 # units: kg * pixels² / s² (joules in pixel units)

print(f"Total work done during squeeze = {work:.4f} (pixel-based units)")

# -------------------------------
# 13. IDENTIFY THE BRAKING PHASE (maximum negative power peak)
# -------------------------------
# Braking phase is when the eyelid muscles actively decelerate the lid.
# This corresponds to the most negative value of power.
# Find index where power is minimum (most negative)
min_power_idx = np.argmin(power)
min_power_time = timestamps[min_power_idx]
min_power_value = power[min_power_idx]

print(f"Braking phase (max negative power) at time = {min_power_time:.4f} seconds")
print(f"Peak negative power = {min_power_value:.4f} (pixel-based units)")

# -------------------------------
# 14. OPTIONAL: PLOT RESULTS (if matplotlib is available)
# -------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Plot distance
    axs[0].plot(timestamps, smoothed_distances, 'b-', label='Smoothed distance')
    axs[0].set_ylabel('Distance (pixels)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocity
    axs[1].plot(timestamps, velocity, 'r-', label='Velocity')
    axs[1].set_ylabel('Velocity (pixels/s)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot power
    axs[2].plot(timestamps, power, 'g-', label='Power')
    axs[2].axvline(x=min_power_time, color='k', linestyle='--', label='Braking peak')
    axs[2].set_ylabel('Power (pixel-based units)')
    axs[2].legend()
    axs[2].grid(True)

    # Plot acceleration
    axs[3].plot(timestamps, acceleration, 'm-', label='Acceleration')
    axs[3].set_xlabel('Time (seconds)')
    axs[3].set_ylabel('Acceleration (pixels/s²)')
    axs[3].legend()
    axs[3].grid(True)

    plt.suptitle('Eyelid Squeeze Kinetics (240 FPS)')
    plt.tight_layout()
    plt.show()

except ImportError:
    print("Matplotlib not installed – skipping plots.")

# -------------------------------
# 15. OUTPUT FEATURE VECTOR (for AI classifier)
# -------------------------------
# The important features for authentication:
# - Full velocity profile (normalized to fixed length, e.g., 100 points)
# - Peak negative power (braking magnitude)
# - Time of braking relative to start
# - Total work
# - Peak velocity and acceleration
# We'll print a simple feature vector.
peak_velocity = np.max(np.abs(velocity))
peak_acceleration = np.max(np.abs(acceleration))
start_idx = 0
end_idx = len(timestamps)-1
duration = timestamps[end_idx] - timestamps[start_idx]

print("\n=== Feature Vector for this Squeeze ===")
print(f"Duration (s): {duration:.4f}")
print(f"Peak velocity (pixels/s): {peak_velocity:.2f}")
print(f"Peak acceleration (pixels/s²): {peak_acceleration:.2f}")
print(f"Total work (pixel-based J): {work:.4f}")
print(f"Braking time (s): {min_power_time:.4f}")
print(f"Braking power (min): {min_power_value:.4f}")
