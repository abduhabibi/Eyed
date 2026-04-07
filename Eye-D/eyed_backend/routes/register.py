# routes/register.py
# API endpoint for user registration (enrollment).
# Expects: multipart/form-data with 'name' (string) and 'video' (file).
# Optionally: multiple videos for better enrollment.
# Returns: user ID and model metadata.

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from routes.data import RegisterResponse, extract_features_from_video
from db_handler import add_user, add_feature_vector
from train import train_model_from_vectors   # We'll define this helper below
import tempfile
import os
import numpy as np
from typing import List

router = APIRouter(prefix="/register", tags=["Enrollment"])

# -------------------------------
# 1. TRAINING HELPER (calls your train.py logic)
# -------------------------------
def train_model_from_vectors(feature_vectors: List[np.ndarray], user_name: str, model_save_path: str):
    """
    Takes a list of feature vectors (numpy arrays) from enrollment videos,
    trains a Gaussian Mixture Model (same as train.py), and saves model.pkl.
    Returns the feature dimension and the trained model path.
    """
    from sklearn.mixture import GaussianMixture
    import pickle

    if len(feature_vectors) < 3:
        raise ValueError("Need at least 3 enrollment videos.")

    X_train = np.vstack(feature_vectors)
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_train)
    train_scores = gmm.score_samples(X_train)
    threshold = np.mean(train_scores) - 2 * np.std(train_scores)

    model_data = {
        'gmm': gmm,
        'threshold': threshold,
        'feature_dim': X_train.shape[1],
        'n_samples': X_train.shape[0]
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    return X_train.shape[1]

# -------------------------------
# 2. REGISTRATION ENDPOINT
# -------------------------------
@router.post("/", response_model=RegisterResponse)
async def register_user(
    name: str = Form(...),
    videos: List[UploadFile] = File(...)   # Accept multiple videos (e.g., 5-10)
):
    """
    Enroll a new user with one or more videos of eyelid squeezes.
    - name: unique identifier for the user.
    - videos: list of video files (MP4, AVI, etc.).
    """
    if len(videos) < 3:
        raise HTTPException(status_code=400, detail="At least 3 videos required for enrollment.")

    feature_vectors = []
    # Process each uploaded video
    for idx, video in enumerate(videos):
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Extract feature vector from this video
            vec, dim = extract_features_from_video(tmp_path)
            feature_vectors.append(vec)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed for video {idx+1}: {str(e)}")
        finally:
            os.unlink(tmp_path)   # clean up

    # Train the model on collected vectors
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}_model.pkl")
    try:
        feature_dim = train_model_from_vectors(feature_vectors, name, model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store user in database
    user_id = add_user(name=name, model_path=model_path, feature_dim=feature_dim)
    if user_id == -1:
        raise HTTPException(status_code=500, detail="Failed to save user to database.")

    # Store each enrollment feature vector in the database (for future incremental updates)
    for vec in feature_vectors:
        add_feature_vector(user_id, vec, vector_type='enrollment')

    return RegisterResponse(
        user_id=user_id,
        name=name,
        message=f"User {name} enrolled successfully with {len(videos)} videos.",
        feature_dim=feature_dim
    )
