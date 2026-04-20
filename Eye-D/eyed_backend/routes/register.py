# routes/register.py
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database.db_handler import DBHandler
from feature_extraction.extract_work_power import extract_total_work

router = APIRouter()
db = DBHandler()

@router.post("/register/")
async def register_user(
    name: str = Form(...),
    videos: list[UploadFile] = File(...),
    # Additional fields will be captured via Form() automatically
):
    if len(videos) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 squeeze videos required.")

    work_values = []
    temp_dir = f"/tmp/register_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for i, video in enumerate(videos):
            file_path = os.path.join(temp_dir, f"squeeze_{i}.webm")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            work = extract_total_work(file_path)
            if work is None:
                raise HTTPException(status_code=400, detail=f"Video {i+1} failed feature extraction.")
            work_values.append(work)

        # Use average work as template
        template_work = sum(work_values) / len(work_values)

        # Collect metadata from form fields (exclude 'name' and 'videos')
        # In FastAPI, you can access all form fields via request.form(), but we'll keep it simple:
        # The frontend sends additional fields as Form data. We'll parse them manually.
        # For simplicity, we'll just store what we can get from the request.
        # In a real app, you'd iterate over all form keys except reserved ones.
        metadata = {}
        # (This part can be expanded as needed)

        user_id = db.insert_user(name, template_work, metadata)
        return {"user_id": user_id, "template_work": template_work}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
