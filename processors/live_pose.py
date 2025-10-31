# processors/live_pose.py
# -*- coding: utf-8 -*-
"""
Realtime YOLOv8-Pose wrapper for live webcam streaming in Streamlit.
- Auto device pick (cuda -> mps -> cpu)
- Mirror preview option
- Draw skeleton on frame
- Return raw keypoints when needed
- Helper to convert BGR ndarray -> av.VideoFrame for streamlit-webrtc

Dependencies (already in requirements):
  ultralytics, numpy, opencv-python, av
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None

try:
    import av
except Exception as e:  # pragma: no cover
    av = None


# ──────────────────────────────────────────────────────────────────────────────
# Drawing config
KP_RADIUS = 4
KP_THICK = 2
LINE_THICK = 2
KP_COLOR = (0, 255, 0)       # BGR green
LIMB_COLOR = (255, 0, 0)     # BGR blue

# YOLOv8/COCO 17-keypoint skeleton pairs (index based)
# 0:nose 1:eye_l 2:eye_r 3:ear_l 4:ear_r 5:shldr_l 6:shldr_r
# 7:elbow_l 8:elbow_r 9:wrist_l 10:wrist_r 11:hip_l 12:hip_r
# 13:knee_l 14:knee_r 15:ankle_l 16:ankle_r
COCO_PAIRS = [
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
    (5, 6),                # shoulders
    (11, 12),              # hips
    (5, 11), (6, 12),      # torso diagonals
    (0, 5), (0, 6)         # head to shoulders
]


def _pick_device(explicit: Optional[str] = None) -> str:
    """
    Choose best available device.
    explicit in {"cuda","mps","cpu"} has highest priority when valid.
    """
    if explicit:
        if explicit == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        if explicit == "mps" and torch is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LivePose:
    """
    Wrapper around YOLOv8-Pose for live inference.

    Usage:
        lp = LivePose("yolov8n-pose.pt", conf=0.35)
        vis, kps = lp.infer_and_draw(bgr, mirror=True, return_kps=True)
    """

    def __init__(
        self,
        yolo_weights: str = "yolov8n-pose.pt",
        conf: float = 0.35,
        device: Optional[str] = None,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Please `pip install ultralytics`."
            )

        self.device = _pick_device(device)
        self.conf = float(conf)

        # Resolve weights path relative to project root if a local file name is given
        if not os.path.isabs(yolo_weights) and not os.path.exists(yolo_weights):
            # try project root
            maybe_local = os.path.join(os.getcwd(), yolo_weights)
            if os.path.exists(maybe_local):
                yolo_weights = maybe_local

        self.model = YOLO(yolo_weights)

        # Some ultralytics versions accept .to(device) for preloading
        try:
            if hasattr(self.model.model, "to"):
                self.model.model.to(self.device)
        except Exception:
            pass

        # Small warmup makes first frame smoother (optional)
        self._warm = False

    # ────────────────────────────────────────────────────────────────────── utils
    @staticmethod
    def _draw_kps_and_limbs(
        canvas: np.ndarray,
        kps_xy: np.ndarray,
        kp_radius: int = KP_RADIUS,
        kp_color: Tuple[int, int, int] = KP_COLOR,
        limb_color: Tuple[int, int, int] = LIMB_COLOR,
        thickness: int = LINE_THICK,
    ) -> None:
        """Draw keypoints and skeleton lines onto BGR canvas in-place."""
        if kps_xy is None or len(kps_xy) == 0:
            return

        # Draw points
        for (x, y) in kps_xy:
            if x > 0 and y > 0:
                cv2.circle(canvas, (int(x), int(y)), kp_radius, kp_color, KP_THICK, cv2.LINE_AA)

        # Draw limbs
        for a, b in COCO_PAIRS:
            xa, ya = kps_xy[a]
            xb, yb = kps_xy[b]
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), limb_color, thickness, cv2.LINE_AA)

    # ───────────────────────────────────────────────────────────── inference API
    def infer_and_draw(
        self,
        bgr: np.ndarray,
        mirror: bool = True,
        return_kps: bool = False,
    ):
        """
        Run a single-frame pose inference and draw skeleton on a copy of the frame.

        Args:
            bgr: np.ndarray in BGR color (H, W, 3)
            mirror: if True, flip horizontally before drawing (mirror preview)
            return_kps: if True, also return first-person keypoints ndarray (N,2)

        Returns:
            vis_only: np.ndarray (if return_kps=False)
            or (vis, kps): Tuple[np.ndarray, Optional[np.ndarray]]
        """
        if bgr is None or bgr.size == 0:
            out = np.zeros((480, 640, 3), np.uint8)
            return (out, None) if return_kps else out

        frame = bgr.copy()
        if mirror:
            frame = cv2.flip(frame, 1)

        # First-call warmup to avoid latency spikes (optional)
        if not self._warm:
            try:
                _ = self.model.predict(
                    source=np.zeros_like(frame),
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                    stream=False,
                    imgsz=max(320, max(frame.shape[:2]) // 2),
                )
            except Exception:
                pass
            self._warm = True

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
            stream=False,
        )

        vis = frame
        kps_out = None

        try:
            if results:
                res = results[0]
                # res.keypoints.xy: (num_person, 17, 2)
                if hasattr(res, "keypoints") and res.keypoints is not None:
                    kps = res.keypoints.xy
                    # move to numpy
                    if hasattr(kps, "cpu"):
                        kps = kps.cpu().numpy()
                    if kps is not None and len(kps) > 0:
                        # take first person
                        person = kps[0]
                        kps_out = person.copy()
                        self._draw_kps_and_limbs(vis, person)
        except Exception:
            # Be fail-tolerant; still return a frame
            kps_out = None

        return (vis, kps_out) if return_kps else vis

    # ───────────────────────────────────────────────────────── frame conversion
    @staticmethod
    def avframe_from_bgr(bgr: np.ndarray):
        """
        Convert BGR ndarray to av.VideoFrame (RGB) for streamlit-webrtc callback.
        """
        if av is None:
            raise ImportError("`av` package is required for streamlit-webrtc video callbacks.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        return frame

    # ────────────────────────────────────────────────────────────────── misc
    def close(self):
        """Optional explicit cleanup."""
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                # Move back to CPU to free GPU/MPS memory if desired
                self.model.model.to("cpu")
        except Exception:
            pass
