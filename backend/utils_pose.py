#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils_pose.py
-------------
Unified utilities to convert YOLOv8-Pose keypoints → pose embedding vectors.

This MUST match build_pose_embeddings.py so that query vectors
are compatible with stored pose_embeddings.npy.
"""

import numpy as np


def encode_keypoints_to_pose_vector(
    keypoints_xy: np.ndarray,   # shape: (17, 2)
    visibility: np.ndarray,     # shape: (17,)
) -> np.ndarray:
    """
    Convert YOLOv8 keypoints → a normalized pose embedding vector.

    Steps:
      1. Normalize XY to [0,1] by image width/height (caller must pre-normalize)
      2. Use visibility as mask
      3. Compute limb angles
      4. Flatten into 1D feature vector
    """

    # ----- 1) Normalize XY (expected by build_pose_embeddings)
    xy = keypoints_xy.astype("float32")
    vis = visibility.astype("float32")

    # Flatten XY
    xy_flat = xy.reshape(-1)  # (34,)

    # ----- 2) visibility mask
    mask = vis.reshape(-1)  # (17,)

    # ----- 3) Compute simple limb angles
    def angle(p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        return np.degrees(np.arctan2(vy, vx))

    # Example pairs (shoulder-left → elbow-left, etc.)
    k = xy
    angle_pairs = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
    ]
    angs = []
    for a, b in angle_pairs:
        p1 = k[a] if vis[a] > 0.1 else None
        p2 = k[b] if vis[b] > 0.1 else None
        angs.append(angle(p1, p2))
    angs = np.array(angs, dtype="float32")  # (4,)

    # ----- 4) Concatenate all
    feat = np.concatenate([
        xy_flat,      # 34
        mask,         # 17
        angs,         # 4
    ]).astype("float32")  # total = 34+17+4 = 55 dims

    return feat  # (55,)
