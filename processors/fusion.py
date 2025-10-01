# processors/fusion.py
import os
import cv2
import numpy as np
from PIL import Image

def read_rgb(x) -> np.ndarray:
    """
    Robust image reader:
    - path string -> cv2.imread (BGR) -> RGB
    - PIL.Image.Image -> to RGB ndarray
    - numpy array (BGR/RGB/GRAY) -> normalize to RGB
    """
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # heuristic: assume BGR and convert to RGB
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        raise ValueError("Unsupported ndarray image shape")
    if isinstance(x, Image.Image):
        return np.array(x.convert("RGB"))
    if isinstance(x, str):
        if not os.path.exists(x):
            raise FileNotFoundError(f"Image path does not exist: {x}")
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imread failed to load: {x}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise TypeError("read_rgb expects path/PIL/numpy image")

def save_rgb(img: np.ndarray, path: str):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ---- Color transfer (LAB mean/std) ----
def color_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype("float32")
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype("float32")
    src_mean, src_std = cv2.meanStdDev(src_lab)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    src_norm = (src_lab - src_mean.reshape(1,1,3)) / (src_std.reshape(1,1,3) + 1e-6)
    transferred = src_norm * ref_std.reshape(1,1,3) + ref_mean.reshape(1,1,3)
    transferred = np.clip(transferred, 0, 255).astype("uint8")
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)

# ---- Texture blend (edge overlay) ----
def texture_blend(src: np.ndarray, ref: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(ref_gray, 80, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(src, 1.0, edges, alpha, 0)

# ---- Full artistic fusion ----
def fuse_artistic(participant_path: str, match_path: str, out_path: str = None) -> np.ndarray:
    src = read_rgb(participant_path)
    ref = read_rgb(match_path)
    ref = cv2.resize(ref, (src.shape[1], src.shape[0]))
    step1 = color_transfer(src, ref)
    step2 = texture_blend(step1, ref, alpha=0.3)
    if out_path:
        save_rgb(step2, out_path)
    return step2
