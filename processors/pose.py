from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

mp_pose = mp.solutions.pose

@dataclass
class PoseResult:
    ok: bool
    keypoints: Optional[np.ndarray]  # (N,2) in [0,1] bbox-normalized
    bbox: Optional[Tuple[int,int,int,int]]  # x,y,w,h

def _bbox_from_keypoints(kps_xy: np.ndarray):
    xs, ys = kps_xy[:,0], kps_xy[:,1]
    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

def extract_pose(pil_img: Image.Image) -> PoseResult:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not res.pose_landmarks:
        return PoseResult(False, None, None)
    pts = []
    for lm in res.pose_landmarks.landmark:
        pts.append([lm.x * w, lm.y * h])
    pts = np.array(pts, dtype=np.float32)  # (33,2) pixel
    x, y, bw, bh = _bbox_from_keypoints(pts)
    bw = max(bw, 1); bh = max(bh, 1)
    pts_norm = np.stack([(pts[:,0]-x)/bw, (pts[:,1]-y)/bh], axis=1)  # [0,1]
    return PoseResult(True, pts_norm.astype(np.float32), (x, y, bw, bh))

def pose_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """a,b: (N,2) in [0,1] → 余弦相似度"""
    if a is None or b is None: return 0.0
    va = a.reshape(-1).astype(np.float32); vb = b.reshape(-1).astype(np.float32)
    na = np.linalg.norm(va) + 1e-8; nb = np.linalg.norm(vb) + 1e-8
    return float(np.dot(va, vb) / (na * nb))
