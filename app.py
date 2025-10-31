# processors/pose.py
# ------------------------------------------------------------
# Minimal, app-compatible pose module:
# - make_pose_detector(model_path="yolov8n-pose.pt", device="auto")
# - PoseEstimator.extract_keypoints_pil(pil_img, detector) -> np.ndarray | None
# - draw_skeleton(image, keypoints) (optional helper)
# If ultralytics YOLOv8-Pose isn't available, gracefully falls back to a Dummy.
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw

# Try to import ultralytics for YOLOv8-Pose
_HAS_YOLO = False
try:
    from ultralytics import YOLO  # type: ignore
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


# --------------------------- Detector wrappers ---------------------------

class _DummyPose:
    """Fallback detector that always returns no keypoints."""
    def __init__(self) -> None:
        pass

    def __call__(self, img_bgr: np.ndarray):
        # mimic ultralytics outputs structure minimally
        class _R:
            keypoints = None
        return [_R()]


class _YOLOPose:
    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model = YOLO(model_path)
        # ultralytics uses .to() via model for device when predicting; we pass device in predict

        # COCO order expected (ultralytics uses 17 kp COCO by default)
        self.num_kp = 17
        self.device = device

    def __call__(self, img_bgr: np.ndarray):
        # ultralytics accepts numpy BGR/RGB; device forwarded via args
        return self.model.predict(
            source=img_bgr,
            device=self.device if self.device != "auto" else None,
            verbose=False,
            stream=False,
        )


def make_pose_detector(
    model_path: Optional[str] = "yolov8n-pose.pt",
    device: str = "auto",
):
    """
    Create a pose detector. Signature is intentionally PARAM-OPTIONAL
    so app.py / retrieval.py can call make_pose_detector() with no args.

    If ultralytics is available and model exists, returns YOLO wrapper.
    Otherwise returns a Dummy detector (no keypoints).
    """
    if _HAS_YOLO and model_path:
        mp = Path(model_path)
        if not mp.exists():
            # also try relative to project root
            alt = Path.cwd() / model_path
            if alt.exists():
                mp = alt
        if mp.exists():
            try:
                return _YOLOPose(str(mp), device=device)
            except Exception:
                pass
    # fallback
    return _DummyPose()


# --------------------------- Keypoint extractor ---------------------------

class PoseEstimator:
    """
    Static helpers to extract & normalize keypoints.
    Returned format: np.ndarray of shape (17, 2) with coordinates normalized to [0,1]
    relative to image width/height. If not found, return None.
    """

    @staticmethod
    def _to_numpy(img: Image.Image) -> np.ndarray:
        # PIL RGB -> BGR (ultralytics不强制，但我们统一BGR)
        arr = np.array(img.convert("RGB"))
        return arr[:, :, ::-1].copy()

    @staticmethod
    def extract_keypoints_pil(pil_img: Image.Image, detector: Any) -> Optional[np.ndarray]:
        """
        Returns 17x2 normalized keypoints or None.
        """
        if detector is None:
            return None

        img_bgr = PoseEstimator._to_numpy(pil_img)
        try:
            preds = detector(img_bgr)
        except Exception:
            return None

        # ultralytics: list of Results; each has .keypoints with .data (n,17,3) (x,y,conf)
        kp = None
        if preds and hasattr(preds[0], "keypoints") and preds[0].keypoints is not None:
            # pick the first person w/ highest confidence avg
            data = preds[0].keypoints.data  # tensor [n, 17, 3]
            try:
                import torch  # local import
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
            except Exception:
                data = np.array(data)

            if data.ndim == 3 and data.shape[1] >= 17:
                # choose the instance with highest mean conf
                conf_mean = data[:, :, 2].mean(axis=1)
                idx = int(conf_mean.argmax())
                xy = data[idx, :17, :2]  # (17,2)
                kp = xy

        if kp is None:
            return None

        # normalize to [0,1] by width/height
        h, w = img_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return None
        kp_norm = kp.astype(np.float32)
        kp_norm[:, 0] /= float(w)
        kp_norm[:, 1] /= float(h)
        return kp_norm  # (17,2)


# --------------------------- Visualization (optional) ---------------------------

_COCO_PAIRS: List[Tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12),  # shoulders & hips
    (0, 5), (0, 6), (5, 11), (6, 12)  # torso-ish links
]

def draw_skeleton(pil_img: Image.Image, keypoints: np.ndarray, radius: int = 4) -> Image.Image:
    """
    Draw COCO skeleton onto a copy of the image. keypoints must be absolute pixel coords
    or normalized (0~1). If normalized (<=1), will auto-scale by image size.
    """
    out = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    w, h = out.size
    pts = keypoints.copy().astype(float)
    # auto-detect normalization
    if pts.max() <= 1.0 + 1e-6:
        pts[:, 0] *= w
        pts[:, 1] *= h

    # draw lines
    for a, b in _COCO_PAIRS:
        if a < len(pts) and b < len(pts):
            ax, ay = pts[a]
            bx, by = pts[b]
            draw.line([(ax, ay), (bx, by)], width=3, fill=(0, 255, 0))

    # draw joints
    for x, y in pts:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return out


__all__ = ["make_pose_detector", "PoseEstimator", "draw_skeleton"]
