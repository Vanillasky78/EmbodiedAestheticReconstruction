# main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model.pose_matcher import PoseMatcher
from config import Settings
from typing import List
import numpy as np
import uvicorn
import base64
from io import BytesIO
from PIL import Image
import os

# Load configuration and initialize the matcher
settings = Settings()
matcher = PoseMatcher(
    data_root=settings.data_root,
    clip_model_name=settings.clip_model_name,
    clip_pretrained=settings.clip_pretrained,
    topk=settings.topk_default,
    use_dual=settings.use_dual_view,
    device=settings.device_override,
    allow_museums=settings.allow_museums,
    verbose=settings.verbose
)

# Initialize FastAPI application
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match")
async def match_image(
    museum: str = Form(...),
    topk: int = Form(3),
    file: UploadFile = File(...),
):
    """Match uploaded image file to artworks."""
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = matcher.match(image=image, museum=museum, topk=topk)
    return results

@app.post("/match_b64")
async def match_b64(
    museum: str = Form(...),
    topk: int = Form(3),
    b64_image: str = Form(...),
):
    """Match base64-encoded image (used by Streamlit frontend)."""
    image_data = base64.b64decode(b64_image.split(",")[-1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    results = matcher.match(image=image, museum=museum, topk=topk)
    return results

@app.get("/metadata/{museum}/{filename}")
async def get_metadata(museum: str, filename: str):
    """Fetch metadata for a matched artwork."""
    meta = matcher.get_metadata(museum, filename)
    return meta

@app.get("/list_museums")
async def list_museums():
    """Return a list of all available museums in the index."""
    return matcher.get_museum_list()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
