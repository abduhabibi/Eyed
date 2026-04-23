import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database.db_handler import DBHandler

router = APIRouter()
db = DBHandler()

ACCEPTANCE_THRESHOLD = 0.20

@router.post("/verify/")
async def verify_user(
    username: str = Form(...),
    video: UploadFile = File(...)
):
    temp_dir = f"/tmp/verify_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, "verify.webm")

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # TODO: Replace with actual feature extraction once scipy is available
        # For now, use filename length as a simple metric
        work = len(video.filename)

        stored_work = db.get_user_template(username)
        if stored_work is None:
            raise HTTPException(status_code=404, detail="User not found.")

        diff = abs(work - stored_work) / max(stored_work, 1e-6)
        accepted = diff <= ACCEPTANCE_THRESHOLD
        score = 1.0 - min(diff, 1.0)

        return {
            "accepted": accepted,
            "score": score,
            "extracted_work": work,
            "template_work": stored_work,
            "threshold": ACCEPTANCE_THRESHOLD
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

