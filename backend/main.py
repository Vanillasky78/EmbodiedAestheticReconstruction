#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backend API for Hybrid CLIP + Pose Matching
-------------------------------------------

Pipeline:
  Receive an uploaded image →
    CLIP image encoder →
    YOLOv8-Pose keypoints → pose vector →
    Hybrid matcher (CLIP + pose + value/portrait/masterpiece priors) →
    Return top-k matched artworks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import io
import numpy as np
from PIL import Image

import torch
import open_clip
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from .model.pose_matcher import PoseMatcher
from .utils_pose import encode_keypoints_to_pose_vector
from .config import Settings


# ---------------------------------------------------------
# Paths & global settings
# ---------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent          # .../backend
PROJECT_ROOT = BACKEND_ROOT.parent                      # repo root: EmbodiedAestheticReconstruction/

settings = Settings()                                   # uses config.py

DATA_ROOT = settings.data_root.parent                   # .../data
DEFAULT_MUSEUM = settings.default_museum                # usually "mixed"

# For now we always use the global mixed index.
MUSEUM_PATH = settings.data_root                        # .../data/mixed


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="EAR Hybrid Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[INFO] Using device: {device}")


# ---------------------------------------------------------
# Load CLIP (once)
# ---------------------------------------------------------
clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
    device=device,
)
clip_model.eval()


# ---------------------------------------------------------
# Load YOLOv8-Pose (once)
# ---------------------------------------------------------
# Prefer the weight at repo root; you also have another copy under frontend/.
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n-pose.pt"
if not YOLO_MODEL_PATH.exists():
    # fallback to frontend/yolov8n-pose.pt
    alt = PROJECT_ROOT / "frontend" / "yolov8n-pose.pt"
    if alt.exists():
        YOLO_MODEL_PATH = alt

print(f"[INFO] Loading YOLO pose model: {YOLO_MODEL_PATH}")
pose_model = YOLO(str(YOLO_MODEL_PATH))


# ---------------------------------------------------------
# Load Hybrid Matcher (global mixed index)
# ---------------------------------------------------------
print(f"[INFO] Loading PoseMatcher with data root: {MUSEUM_PATH}")
matcher = PoseMatcher(
    museum_dir=MUSEUM_PATH,
    pose_weight=settings.pose_weight,
    topk_default=10,
)


# =========================================================
# Helper: extract YOLO keypoints from image
# =========================================================
def extract_keypoints(pil: Image.Image):
    """
    Run YOLOv8-Pose on a PIL image and return:
        xy_norm: (17, 2)  normalized to [0, 1]
        conf:    (17,)    keypoint confidence
    Returns None if no person / keypoints found.
    """
    rgb = np.array(pil.convert("RGB"))
    results = pose_model.predict(rgb, imgsz=640, device=str(device), verbose=False)

    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None

    kpts = results[0].keypoints
    xy = kpts.xy[0].cpu().numpy()    # (17, 2)
    conf = kpts.conf[0].cpu().numpy()  # (17,)

    h, w = rgb.shape[:2]
    xy_norm = xy.copy()
    xy_norm[:, 0] /= float(w)
    xy_norm[:, 1] /= float(h)

    return xy_norm, conf


# =========================================================
# API endpoint
# =========================================================
@app.post("/match")
async def match_image(
    image: UploadFile,
    museum: Optional[str] = Form(DEFAULT_MUSEUM),
    topk: Optional[int] = Form(5),
    mode: Optional[str] = Form("default"),  # "default", "portrait_only", etc.
):
    """
    Main matching endpoint.

    Parameters sent from frontend (Streamlit):
      - image: captured frame (JPEG/PNG)
      - museum: currently ignored (we always use global mixed index)
      - topk: number of results to return
      - mode: matching mode for PoseMatcher (default / portrait_only / ...)
    """
    # ------------------- load image -------------------
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    # ------------------- CLIP query vec ----------------
    img_tensor = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=str(device),
        enabled=(device.type in ["cuda", "mps"]),
    ):
        feat = clip_model.encode_image(img_tensor)
    q_clip = feat.float().cpu().numpy()
    q_clip = q_clip / (np.linalg.norm(q_clip, axis=1, keepdims=True) + 1e-12)

    # ------------------- Pose query vec ----------------
    kp = extract_keypoints(pil)
    if kp is not None:
        xy_norm, conf = kp
        pose_vec_1d = encode_keypoints_to_pose_vector(xy_norm, conf).astype("float32")
        q_pose = pose_vec_1d[None, :]
        q_pose = q_pose / (np.linalg.norm(q_pose, axis=1, keepdims=True) + 1e-12)
    else:
        q_pose = None

    # ------------------- Hybrid match ------------------
    results = matcher.match(
        clip_query_vec=q_clip,
        pose_query_vec=q_pose,
        topk=topk,
        mode=mode or "default",
    )

    return {
        "results": results,
        "museum": DEFAULT_MUSEUM,
        "mode": mode,
    }
