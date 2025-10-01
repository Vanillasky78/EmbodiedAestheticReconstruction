# processors/fusion.py
import cv2, numpy as np
from PIL import Image

def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_rgb(img: np.ndarray, path: str):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ---- Color transfer ----
def color_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype("float32")
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype("float32")
    src_mean, src_std = cv2.meanStdDev(src_lab); ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    src_norm = (src_lab-src_mean.reshape(1,1,3))/(src_std.reshape(1,1,3)+1e-6)
    transferred = src_norm*ref_std.reshape(1,1,3)+ref_mean.reshape(1,1,3)
    transferred = np.clip(transferred,0,255).astype("uint8")
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)

# ---- Texture blend ----
def texture_blend(src: np.ndarray, ref: np.ndarray, alpha: float=0.4) -> np.ndarray:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(ref_gray, 80, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(src,1.0,edges,alpha,0)

# ---- Full artistic fusion ----
def fuse_artistic(participant_path: str, match_path: str, out_path: str=None) -> np.ndarray:
    src, ref = read_rgb(participant_path), read_rgb(match_path)
    ref = cv2.resize(ref,(src.shape[1],src.shape[0]))
    step1 = color_transfer(src,ref)
    step2 = texture_blend(step1,ref,alpha=0.3)
    if out_path: save_rgb(step2,out_path)
    return step2
