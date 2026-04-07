# routes/data.py
# Shared data models and video-to-feature conversion logic for the Eye-D API.
# Uses Pydantic for request/response validation.
# Provides a function `extract_features_from_video(video_path)` that:
#   - Runs the five feature extraction scripts sequentially.
#   - Loads their output .npy files.
#   - Concatenates them into a single feature vector.
#   - Returns the vector and its dimension.

import os
import subprocess
import numpy as np
from pydantic import BaseModel
from typing import List, Optional

# -------------------------------
# 1. REQUEST/RESPONSE MODELS
# -------------------------------

class RegisterRequest(BaseModel):
    """Expected JSON body for user registration."""
    name: str                     # User's unique display name
    # We'll also accept multiple video files via form-data, not JSON.
    # This model is for metadata only.

class RegisterResponse(BaseModel):
    """Response after successful registration."""
    user_id: int
    name: str
    message: str
    feature_dim: int

class VerifyRequest(BaseModel):
    """Metadata for verification (actual video sent as file)."""
    user_id: Optional[int] = None
    username: Optional[str] = None   # either user_id or username

class VerifyResponse(BaseModel):
    """Response after verification attempt."""
    accepted: bool
    score: float
    threshold: float
    message: str

# -------------------------------
# 2. VIDEO TO FEATURE VECTOR CONVERSION
# -------------------------------
# Paths to your feature extraction scripts (adjust as needed)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTORS = {
    'kinetic': os.path.join(SCRIPT_DIR, 'extract_work_power.py'),
    'path_3d': os.path.join(SCRIPT_DIR, 'extract_3d_path.py'),
    'main_sequence': os.path.join(SCRIPT_DIR, 'extract_main_sequence.py'),
    'wave': os.path.join(SCRIPT_DIR, 'extract_wave_closure.py'),
    'wrinkle': os.path.join(SCRIPT_DIR, 'extract_wrinkle_sequence.py')
}

def extract_features_from_video(video_path: str) -> tuple:
    """
    Takes a video file path, runs all extraction scripts, and returns a
    concatenated feature vector (numpy array) and its dimension.
    Each extraction script is expected to output a .npy file in the same directory.
    We'll collect those files and combine them.
    """
    # Create a temporary working directory for this video's outputs
    temp_dir = os.path.join(SCRIPT_DIR, "temp_features")
    os.makedirs(temp_dir, exist_ok=True)

    # Because extraction scripts currently read from fixed INPUT_VIDEO_PATH,
    # we need to either modify them to accept arguments or symlink the video.
    # For simplicity, we'll copy the video to a known location each time.
    # In production, you should refactor the scripts to accept command-line arguments.
    fixed_input = os.path.join(SCRIPT_DIR, "squeeze_video.avi")
    # Copy or move the uploaded video to the fixed path
    import shutil
    shutil.copy2(video_path, fixed_input)

    # Run each extraction script
    for name, script_path in EXTRACTORS.items():
        if os.path.exists(script_path):
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            if result.returncode != 0:
                print(f"Error running {name} extractor: {result.stderr}")
                # Continue with others? For now, raise exception.
                raise RuntimeError(f"Extractor {name} failed: {result.stderr}")
        else:
            print(f"Warning: {script_path} not found. Skipping {name}.")

    # Now load all generated .npy files (named as per the scripts)
    feature_files = [
        os.path.join(SCRIPT_DIR, "feature1_kinetic.npy"),
        os.path.join(SCRIPT_DIR, "feature2_main_sequence.npy"),
        os.path.join(SCRIPT_DIR, "feature3_3d_path.npy"),
        os.path.join(SCRIPT_DIR, "feature4_wrinkle.npy"),
        os.path.join(SCRIPT_DIR, "feature5_wave_closure.npy")
    ]
    vectors = []
    for fpath in feature_files:
        if os.path.exists(fpath):
            vec = np.load(fpath)
            # Ensure 1D array
            vec = vec.flatten()
            vectors.append(vec)
        else:
            print(f"Warning: {fpath} not found. Using zero vector placeholder.")
            # Placeholder: zero vector of expected size (you should know expected dims)
            # For robustness, we'll assume a default dimension if missing.
            vectors.append(np.zeros(1))

    # Concatenate all feature vectors
    full_vector = np.concatenate(vectors)
    return full_vector, len(full_vector)
