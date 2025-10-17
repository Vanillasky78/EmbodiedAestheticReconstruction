"""
similarity.py — compute multi-modal similarity between person input and artworks.
Used in Match-only mode for Embodied Aesthetic Reconstruction.
"""

import numpy as np
from numpy.linalg import norm
import cv2

# --- Pose similarity ---
def pose_similarity(kps1: np.ndarray, kps2: np.ndarray) -> float:
    """Compute cosine similarity between two pose keypoint arrays (normalized)."""
    if kps1 is None or kps2 is None:
        return 0.0
    if len(kps1) != len(kps2):
        n = min(len(kps1), len(kps2))
        kps1, kps2 = kps1[:n], kps2[:n]
    v1 = kps1.flatten()
    v2 = kps2.flatten()
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-8))


# --- Color palette similarity ---
def color_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compare color histograms using correlation (0–1)."""
    if img1 is None or img2 is None:
        return 0.0
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, sim)))


# --- Combine scores ---
def combine_score(clip_s: float, pose_s: float, color_s: float,
                  w_clip: float = 0.6, w_pose: float = 0.25, w_color: float = 0.15) -> float:
    """Weighted combination of different similarity measures."""
    score = w_clip * clip_s + w_pose * pose_s + w_color * color_s
    return float(max(0.0, min(1.0, score)))
