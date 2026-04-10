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
import sys
import tempfile
import shutil
import json
from pydantic import BaseModel
from typing import List, Optional

# ========================================
# 1. REQUEST/RESPONSE MODELS
# ========================================

class RegisterRequest(BaseModel):
    """Expected JSON body for user registration."""
    name: str                     # User's unique display name

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

# ========================================
# 2. VIDEO TO FEATURE VECTOR CONVERSION
# ========================================
# Paths to your feature extraction scripts (adjust as needed)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTORS = {
    'kinetic': os.path.join(SCRIPT_DIR, 'feature_extraction', 'extract_work_power.py'),
    'path_3d': os.path.join(SCRIPT_DIR, 'feature_extraction', 'extract_3d_path.py'),
    'main_sequence': os.path.join(SCRIPT_DIR, 'feature_extraction', 'extract_main_sequence.py'),
    'wave': os.path.join(SCRIPT_DIR, 'feature_extraction', 'extract_wave_closure.py'),
    'wrinkle': os.path.join(SCRIPT_DIR, 'feature_extraction', 'extract_wrinkle_sequence.py')
}

def extract_features_from_video(video_path: str) -> tuple:
    """
    Takes a video file path, runs all extraction scripts, and returns a
    concatenated feature vector (numpy array) and its dimension.
    Each extraction script is expected to output a .npy file in the same directory.
    We'll collect those files and combine them.
    """
    # Per-request isolated workspace to avoid concurrency collisions.
    # Everything (a request-local input copy + all .npy outputs) lives here and is cleaned up.
    base_temp_dir = os.path.join(SCRIPT_DIR, "temp_features")
    os.makedirs(base_temp_dir, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="eyed_", dir=base_temp_dir) as work_dir:
            request_input = os.path.join(work_dir, "squeeze_video.avi")
            shutil.copy2(video_path, request_input)

            # Run each extraction script against the request-local input and output directory
            for name, script_path in EXTRACTORS.items():
                if not os.path.exists(script_path):
                    raise RuntimeError(f"Extractor script missing: {script_path}")

                result = subprocess.run(
                    [sys.executable, script_path, "--input", request_input, "--outdir", work_dir],
                    capture_output=True,
                    text=True,
                    cwd=SCRIPT_DIR,
                    timeout=30
                )
                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    stdout = (result.stdout or "").strip()
                    detail = stderr if stderr else stdout
                    raise RuntimeError(f"Extractor {name} failed: {detail}")

            # Load all generated .npy files from the request-local output directory
            feature_files = [
                os.path.join(work_dir, "feature1_kinetic.npy"),
                os.path.join(work_dir, "feature2_main_sequence.npy"),
                os.path.join(work_dir, "feature3_3d_path.npy"),
                os.path.join(work_dir, "feature4_wrinkle.npy"),
                os.path.join(work_dir, "feature5_wave_closure.npy")
            ]
            vectors = []
            for fpath in feature_files:
                if not os.path.exists(fpath):
                    raise RuntimeError(f"Missing extractor output: {os.path.basename(fpath)}")
                vec = np.load(fpath).flatten()
                vectors.append(vec)

        # Concatenate all feature vectors
        full_vector = np.concatenate(vectors)
        return full_vector, len(full_vector)
    
    except Exception as e:
        raise RuntimeError(f"Feature extraction pipeline failed: {str(e)}")
