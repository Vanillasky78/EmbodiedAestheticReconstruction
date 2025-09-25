# core/pose.py
import numpy as np
from PIL import Image
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

def _normalise_landmarks(xy: np.ndarray) -> np.ndarray:
    if xy.size == 0:
        return xy
    center = xy.mean(axis=0)
    xy_c = xy - center
    scale = np.linalg.norm(xy_c) / (np.sqrt(max(len(xy_c), 1)) or 1)
    scale = max(scale, 1e-6)
    return xy_c / scale

def pose_to_embedding(img: Image.Image) -> np.ndarray:
    if not MP_AVAILABLE:
        return np.zeros(66, dtype=np.float32)  # 33*2
    img_rgb = np.array(img.convert("RGB"))
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return np.zeros(66, dtype=np.float32)
    lm = res.pose_landmarks.landmark
    xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
    xy = _normalise_landmarks(xy)
    return xy.reshape(-1).astype(np.float32)
