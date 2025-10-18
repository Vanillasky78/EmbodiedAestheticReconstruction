# processors/pose.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

# ---- mediapipe (friendly import) ----
try:
    import mediapipe as mp
except Exception:
    raise SystemExit("Install required package: pip install mediapipe")


mp_pose = mp.solutions.pose


@dataclass
class PoseResult:
    """
    Result of single-person pose extraction.

    Attributes:
        ok: whether landmarks were detected.
        keypoints: (N, 2) array in [0,1], bbox-normalized (x,y). None if not ok.
        bbox: (x, y, w, h) in pixel coordinates on the original image. None if not ok.
    """
    ok: bool
    keypoints: Optional[np.ndarray]          # (N,2) in [0,1] bbox-normalized
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h (pixels)


def _bbox_from_keypoints(kps_xy: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute tight bounding box (x, y, w, h) from pixel keypoints."""
    xs, ys = kps_xy[:, 0], kps_xy[:, 1]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    w = max(int(round(x2 - x1)), 1)
    h = max(int(round(y2 - y1)), 1)
    return int(round(x1)), int(round(y1)), w, h


def extract_pose(pil_img: Image.Image) -> PoseResult:
    """
    Extract human pose landmarks with MediaPipe.

    Returns:
        PoseResult with bbox-normalized keypoints in [0,1].
        If no valid landmarks, returns PoseResult(False, None, None).
    """
    # Convert to RGB numpy array
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]

    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if not res.pose_landmarks:
        return PoseResult(False, None, None)

    # Collect landmarks; filter low-visibility points (keep if visibility is absent or >= 0.3)
    pts: List[Tuple[float, float]] = []
    vis_thresh = 0.3
    for lm in res.pose_landmarks.landmark:
        if (getattr(lm, "visibility", 1.0) or 1.0) >= vis_thresh:
            pts.append((lm.x * w, lm.y * h))

    if len(pts) < 5:  # too few reliable points
        return PoseResult(False, None, None)

    pts_px = np.asarray(pts, dtype=np.float32)  # (N,2) in pixel coords

    # Bounding box & normalization to [0,1] in bbox coordinates
    x, y, bw, bh = _bbox_from_keypoints(pts_px)
    pts_norm = np.stack(((pts_px[:, 0] - x) / bw, (pts_px[:, 1] - y) / bh), axis=1)
    pts_norm = np.clip(pts_norm, 0.0, 1.0).astype(np.float32)

    return PoseResult(True, pts_norm, (x, y, bw, bh))
