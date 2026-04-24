import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database.db_handler import DBHandler

router = APIRouter()
db = DBHandler()

@router.post("/register/")
async def register_user(
    name: str = Form(...),
    videos: list[UploadFile] = File(...),
    **kwargs  # Capture additional form fields (id_number, age, etc)
):
    if len(videos) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 squeeze videos required.")

    temp_dir = f"/tmp/register_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Store videos (for future use)
        for i, video in enumerate(videos):
            file_path = os.path.join(temp_dir, f"squeeze_{i}.webm")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(video.file, f)

        # TODO: Replace with actual feature extraction once scipy is available
        # For now, use a simple hash-based template from the video files
        template_work = sum(len(video.filename) for video in videos) / len(videos)

        # Collect metadata from form fields (exclude 'name' and 'videos')
        metadata = {k: v for k, v in kwargs.items() if k not in ['name', 'videos']}

        user_id = db.insert_user(name, template_work, metadata)
        return {"user_id": user_id, "template_work": template_work}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
