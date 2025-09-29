# processors/pose.py
import numpy as np, cv2
from PIL import Image
import mediapipe as mp
from dataclasses import dataclass

mp_pose = mp.solutions.pose

@dataclass
class PoseResult:
    ok: bool
    keypoints: np.ndarray | None
    bbox: tuple[int,int,int,int] | None

def extract_pose(pil_img: Image.Image) -> PoseResult:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not res.pose_landmarks:
        return PoseResult(False, None, None)
    pts = np.array([[lm.x*w, lm.y*h] for lm in res.pose_landmarks.landmark])
    x1, y1, x2, y2 = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    pts_norm = np.stack([(pts[:,0]-x1)/bw, (pts[:,1]-y1)/bh], axis=1)
    return PoseResult(True, pts_norm, (int(x1), int(y1), int(bw), int(bh)))

def pose_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    va, vb = a.reshape(-1), b.reshape(-1)
    return float(np.dot(va, vb) / (np.linalg.norm(va)*np.linalg.norm(vb)+1e-8))
