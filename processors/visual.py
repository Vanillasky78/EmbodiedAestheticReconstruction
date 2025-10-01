# processors/visual.py
import cv2, numpy as np
from typing import Optional, Tuple
from PIL import Image
import mediapipe as mp

# ---- Pose similarity (Procrustes) ----
def procrustes_pose_score(kps_a: np.ndarray, kps_b: np.ndarray) -> float:
    if kps_a is None or kps_b is None: return 0.0
    if len(kps_a) != len(kps_b):
        n = min(len(kps_a), len(kps_b))
        kps_a, kps_b = kps_a[:n], kps_b[:n]
    A, B = kps_a.astype(np.float32), kps_b.astype(np.float32)
    A -= A.mean(0); B -= B.mean(0)
    A /= (np.linalg.norm(A)+1e-8); B /= (np.linalg.norm(B)+1e-8)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    A2 = A @ R
    err = np.mean(np.linalg.norm(A2 - B, axis=1))
    return float(np.exp(-5.0 * err))

# ---- Face orientation similarity ----
_mp_face = mp.solutions.face_mesh
def _face_angles_from_landmarks(landmarks, w: int, h: int) -> Optional[Tuple[float,float,float]]:
    ids = [33, 263, 1, 152]  # left eye, right eye, nose tip, chin
    try:
        pts = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in ids], dtype=np.float32)
    except Exception:
        return None
    le, re, nose, chin = pts
    eye_vec = re - le
    face_vec = chin - nose
    yaw = np.arctan2(eye_vec[1], eye_vec[0])
    pitch = np.arctan2(face_vec[1], face_vec[0])
    roll = 0.0
    return float(yaw), float(pitch), float(roll)

def face_angle_score(pil_img: Image.Image, pil_ref: Image.Image) -> float:
    def angles(img: Image.Image):
        im = np.array(img.convert("RGB")); h,w = im.shape[:2]
        with _mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            res = fm.process(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if not res.multi_face_landmarks: return None
        return _face_angles_from_landmarks(res.multi_face_landmarks[0].landmark, w,h)
    a1,a2 = angles(pil_img), angles(pil_ref)
    if a1 is None or a2 is None: return 0.0
    diff = np.abs(np.array(a1)-np.array(a2))
    return float(np.exp(-2.0*np.linalg.norm(diff)))

# ---- Silhouette similarity ----
_mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation
def person_mask(pil_img: Image.Image, thr: float=0.5) -> np.ndarray:
    im = np.array(pil_img.convert("RGB"))
    with _mp_selfie(model_selection=1) as seg:
        m = seg.process(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)).segmentation_mask
    return (m>thr).astype(np.uint8)*255

def silhouette_ssim(pil_img: Image.Image, pil_ref: Image.Image) -> float:
    m1,m2 = person_mask(pil_img), person_mask(pil_ref)
    h,w = m1.shape[:2]; m2 = cv2.resize(m2,(w,h))
    m1 = m1.astype(np.float32)/255.0; m2 = m2.astype(np.float32)/255.0
    mse = np.mean((m1-m2)**2)
    return float(np.exp(-4.0*mse))

# ---- Color similarity ----
def color_corr(pil_img: Image.Image, pil_ref: Image.Image) -> float:
    def hist(pil):
        bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
        cv2.normalize(h,h,0,1,cv2.NORM_MINMAX); return h
    h1,h2 = hist(pil_img), hist(pil_ref)
    sim = cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
    return float(np.clip((sim+1)/2,0,1))
