# routes/verify.py
# API endpoint for user verification (authentication).
# Expects: either user_id or username in form-data, plus a video file.
# Returns: accept/reject decision, score, threshold.

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from routes.data import VerifyResponse, extract_features_from_video
from database.db_handler import get_user, log_authentication
import tempfile
import os
import pickle

router = APIRouter(prefix="/verify", tags=["Authentication"])

# -------------------------------
# 1. VERIFICATION ENDPOINT
# -------------------------------
@router.post("/", response_model=VerifyResponse)
async def verify_user(
    video: UploadFile = File(...),
    user_id: int = Form(None),
    username: str = Form(None)
):
    """
    Authenticate a user by comparing the eyelid squeeze in the video
    against the stored model.
    Provide either user_id or username.
    """
    if user_id is None and username is None:
        raise HTTPException(status_code=400, detail="Either user_id or username must be provided.")

    # Retrieve user from database
    user = get_user(user_id=user_id, name=username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract feature vector from the video
        feature_vector, dim = extract_features_from_video(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    # Load the user's model
    if not os.path.exists(user.model_path):
        raise HTTPException(status_code=500, detail="Model file missing.")
    with open(user.model_path, 'rb') as f:
        model_data = pickle.load(f)

    gmm = model_data['gmm']
    threshold = model_data['threshold']

    # Compute log-likelihood
    score = gmm.score_samples(feature_vector.reshape(1, -1))[0]
    accepted = score >= threshold

    # Log the attempt
    log_authentication(user.id, score, threshold, accepted)

    message = "Access granted" if accepted else "Access denied"
    return VerifyResponse(
        accepted=accepted,
        score=score,
        threshold=threshold,
        message=message
    )
