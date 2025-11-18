"""
utils_pose.py — safe pose encoder
---------------------------------

Convert YOLO keypoints → 1D pose feature vector.
Handles:
  - missing keypoints
  - confidence arrays missing
  - zero-length keypoints (no detection)
"""

import numpy as np


def encode_keypoints_to_pose_vector(xy_norm, conf):
    """
    Convert YOLO keypoints (xy_norm, conf) → a fixed-length 1D pose vector.

    Parameters
    ----------
    xy_norm : np.ndarray (N, 2)
        Normalized XY coordinates in [0,1].
    conf : np.ndarray (N,)
        Visibility/confidence for each keypoint.

    Returns
    -------
    vec : np.ndarray or None
        1D pose feature vector (55 dims). None if invalid.
    """

    # -------- Safety: YOLO returned 0 keypoints --------
    if xy_norm is None or len(xy_norm) == 0:
        return None
    if conf is None or len(conf) == 0:
        return None

    # Expected 17 keypoints (COCO format)
    N = xy_norm.shape[0]
    if N < 5:
        # too few keypoints → useless for pose embedding
        return None

    # Build dict: k[i] = (x,y) or None
    k = {}
    vis = conf
    for i in range(N):
        if vis[i] > 0.1:
            k[i] = xy_norm[i]
        else:
            k[i] = None

    # -------- Feature extraction examples --------
    def angle(p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        v = p2 - p1
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def dist(p1, p2):
        if p1 is None or p2 is None:
            return 0.0
        return float(np.linalg.norm(p2 - p1))

    # Example 55-dim handmade embedding
    feats = []

    # Angles between some key joints
    pairs = [
        (1, 2),
        (5, 6),
        (11, 12),
        (7, 8),
    ]

    for a, b in pairs:
        if a < N and b < N:
            feats.append(angle(k[a], k[b]))
        else:
            feats.append(0.0)

    # Distances between some parts
    dist_pairs = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
    ]

    for a, b in dist_pairs:
        if a < N and b < N:
            feats.append(dist(k[a], k[b]))
        else:
            feats.append(0.0)

    # Append raw XY for all available keypoints (flattened)
    for i in range(min(N, 17)):
        p = k[i]
        if p is None:
            feats.extend([0.0, 0.0])
        else:
            feats.extend([float(p[0]), float(p[1])])

    # Pad to 55 dims
    while len(feats) < 55:
        feats.append(0.0)

    return np.array(feats, dtype="float32")
