#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backend API for Hybrid CLIP + Pose Matching
-------------------------------------------
Receives an image → extracts CLIP + YOLO Pose → hybrid match →
returns top-k artworks with metadata priors.
"""

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
import numpy as np
from PIL import Image

import torch
import open_clip
from pathlib import Path

# Relative imports inside backend/
from .model.pose_matcher import PoseMatcher
from .utils_pose import encode_keypoints_to_pose_vector

# YOLOv8-Pose
from ultralytics import YOLO


# ============================================================
# App initialization
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


# ============================================================
# Load CLIP model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

print(f"[INFO] Using device: {device}")

clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=device
)
clip_model.eval()


# ============================================================
# Load YOLOv8-Pose model
# ============================================================

YOLO_MODEL_PATH = ROOT / "frontend" / "yolov8n-pose.pt"
print(f"[INFO] Loading YOLO pose model: {YOLO_MODEL_PATH}")
pose_model = YOLO(str(YOLO_MODEL_PATH))


# ============================================================
# Load Hybrid Matcher (default = mixed index)
# ============================================================

DEFAULT_MUSEUM = "mixed"
matcher = PoseMatcher(DATA_ROOT / DEFAULT_MUSEUM, pose_weight=0.35, topk_default=10)


# ============================================================
# Helper: extract keypoints
# ============================================================

def extract_keypoints(pil: Image.Image):
    """
    Returns normalized keypoints and visibility:
       xy_norm: (17,2) in [0,1]
       vis:     (17,)
    or returns None if no detection.
    """
    rgb = np.array(pil.convert("RGB"))
    res = pose_model.predict(rgb, imgsz=640, device=str(device), verbose=False)

    if len(res) == 0 or res[0].keypoints is None or len(res[0].keypoints) == 0:
        return None

    k = res[0].keypoints
    xy = k.xy[0].cpu().numpy()      # (17,2)
    conf = k.conf[0].cpu().numpy()  # (17,)

    # normalize to [0,1]
    h, w = rgb.shape[:2]
    xy_norm = np.copy(xy)
    xy_norm[:, 0] /= float(w)
    xy_norm[:, 1] /= float(h)

    return xy_norm, conf


# ============================================================
# API: /match
# ============================================================

@app.post("/match")
async def match_image(
    image: UploadFile,
    museum: Optional[str] = Form("mixed"),
    mode: Optional[str] = Form("default"),   # portrait_only / high_value_only / hybrid filters
    topk: Optional[int] = Form(5),
):
    # ---------------------------------------
    # Load image
    # ---------------------------------------
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    # ---------------------------------------
    # Encode CLIP
    # ---------------------------------------
    x = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=str(device), enabled=(device.type in ["cuda", "mps"])
    ):
        feat = clip_model.encode_image(x)

    q_clip = feat.float().cpu().numpy()
    q_clip /= (np.linalg.norm(q_clip, axis=1, keepdims=True) + 1e-12)

    # ---------------------------------------
    # YOLO Pose → pose vector
    # ---------------------------------------
    kp = extract_keypoints(pil)
    if kp is not None:
        xy, vis = kp
        q_pose = encode_keypoints_to_pose_vector(xy, vis).astype("float32")
        q_pose = q_pose[None, :]
        q_pose /= (np.linalg.norm(q_pose, axis=1, keepdims=True) + 1e-12)
    else:
        q_pose = None

    # ---------------------------------------
    # Hybrid match
    # ---------------------------------------
    results = matcher.match(
        clip_query_vec=q_clip,
        pose_query_vec=q_pose,
        topk=topk,
        mode=mode,
    )

    return {
        "results": results,
        "museum": museum,
        "mode": mode,
    }


# ============================================================
# Root test endpoint
# ============================================================

@app.get("/")
def root():
    return {"status": "OK", "matcher": "Hybrid CLIP+Pose backend running"}
