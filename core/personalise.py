# core/personalise.py
import math, numpy as np
from typing import List, Tuple

def personality_vector(traits: List[Tuple[str, float]]) -> np.ndarray:
    return np.array([v for _, v in traits], dtype=np.float32)

def fuse_embeddings(pose_emb: np.ndarray, pers_vec: np.ndarray) -> np.ndarray:
    if pose_emb.size == 0:
        return pose_emb
    reps = int(math.ceil(pose_emb.size / pers_vec.size))
    tiled = np.tile(pers_vec, reps)[: pose_emb.size]
    a = pose_emb / (np.linalg.norm(pose_emb) + 1e-8)
    b = tiled / (np.linalg.norm(tiled) + 1e-8)
    fused = 0.85 * a + 0.15 * b
    return fused / (np.linalg.norm(fused) + 1e-8)
