# processors/similarity.py
"""
Compute multi-modal similarity between a person query and artworks.
Used in Match-only mode for Embodied Aesthetic Reconstruction.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import cv2


# ----------------------------
# Pose similarity
# ----------------------------
def pose_similarity(kps1: Optional[np.ndarray], kps2: Optional[np.ndarray]) -> float:
    """
    Cosine similarity between two bbox-normalized pose keypoint arrays.
    Args:
        kps1, kps2: (N,2) arrays in [0,1]. If shapes differ, use the min(N).
    Returns:
        float in [0,1], 0.0 if either is None.
    """
    if kps1 is None or kps2 is None:
        return 0.0
    if kps1.ndim != 2 or kps2.ndim != 2 or kps1.shape[1] != 2 or kps2.shape[1] != 2:
        return 0.0
    n = min(len(kps1), len(kps2))
    if n <= 0:
        return 0.0
    v1 = kps1[:n].astype(np.float32).reshape(-1)
    v2 = kps2[:n].astype(np.float32).reshape(-1)
    na = np.linalg.norm(v1) + 1e-8
    nb = np.linalg.norm(v2) + 1e-8
    sim = float(np.dot(v1, v2) / (na * nb))
    # cosine can be slightly negative due to noise; clamp to [0,1]
    return max(0.0, min(1.0, sim))


# ----------------------------
# Color similarity (BGR expected)
# ----------------------------
def _hsv_hist(img_bgr: np.ndarray) -> np.ndarray:
    """
    Build a 2D HSV histogram (H,S). Input must be BGR (OpenCV default).
    Returns a L1-normalized flattened histogram (float32).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype("float32")


def color_similarity(img1_bgr: Optional[np.ndarray], img2_bgr: Optional[np.ndarray]) -> float:
    """
    Compare color distributions via Bhattacharyya distance on HSV histograms.
    Args expect BGR images (OpenCV default).
    Returns float in [0,1], where 1 means very similar.
    """
    if img1_bgr is None or img2_bgr is None:
        return 0.0
    try:
        h1 = _hsv_hist(img1_bgr)
        h2 = _hsv_hist(img2_bgr)
        # OpenCV returns distance in [0, +inf); 0 = identical
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        sim = 1.0 - float(dist)
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


# ----------------------------
# Weighted fusion
# ----------------------------
_DEFAULT_W = dict(w_clip=0.6, w_pose=0.3, w_color=0.1)


def combine_score(
    sim_clip: float,
    sim_pose: float,
    sim_color: float,
    meta_bonus: float = 0.0,
    w: Optional[Dict[str, float]] = None,
) -> float:
    """
    Fuse similarities with weights and optional metadata bonus.
    Args:
        sim_clip, sim_pose, sim_color: component similarities in [0,1]
        meta_bonus: small additive term (e.g., +0.02 for public domain)
        w: optional dict overriding weights: {'w_clip','w_pose','w_color'}
    Returns:
        final score in [0,1]
    """
    ww = {**_DEFAULT_W, **(w or {})}
    s = (
        ww.get("w_clip", 0.6) * sim_clip
        + ww.get("w_pose", 0.3) * sim_pose
        + ww.get("w_color", 0.1) * sim_color
        + float(meta_bonus)
    )
    return float(max(0.0, min(1.0, s)))
