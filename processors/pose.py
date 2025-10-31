# processors/pose.py
# ------------------------------------------------------------
# Minimal, app-compatible pose module:
# - make_pose_detector(model_path="yolov8n-pose.pt", device="auto")
# - PoseEstimator.extract_keypoints_pil(pil_img, detector) -> np.ndarray | None
# - draw_skeleton(image, keypoints, ...) -> PIL.Image
# 若未安装 ultralytics 或模型不可用，会优雅降级为 Dummy 检测器（无关键点）。
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw

__version__ = "1.0.0"

# --------------------------- 可选：YOLOv8-Pose ---------------------------
_HAS_YOLO = False
try:
    from ultralytics import YOLO  # type: ignore
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


# --------------------------- Detector wrappers ---------------------------

class _DummyPose:
    """后备检测器：始终返回无关键点，用于缺省/出错时降级。"""
    def __call__(self, img_bgr: np.ndarray):
        class _R:
            keypoints = None
        return [_R()]


class _YOLOPose:
    """Ultralytics YOLOv8-Pose 包装器。"""
    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model = YOLO(model_path)
        self.num_kp = 17  # COCO 17点
        self.device = device

    def __call__(self, img_bgr: np.ndarray):
        # 直接传 numpy 数组给 ultralytics；device 通过参数控制
        return self.model.predict(
            source=img_bgr,
            device=(None if self.device == "auto" else self.device),
            verbose=False,
            stream=False,
        )


def make_pose_detector(
    model_path: Optional[str] = "yolov8n-pose.pt",
    device: str = "auto",
):
    """
    创建姿态检测器。若 YOLOv8-Pose 不可用或模型不存在，则返回 Dummy 检测器。
    签名保持无必填参数，方便 app.py 直接调用 make_pose_detector()。
    """
    if _HAS_YOLO and model_path:
        mp = Path(model_path)
        if not mp.exists():
            # 再尝试以当前工作目录为基准
            alt = Path.cwd() / model_path
            if alt.exists():
                mp = alt
        if mp.exists():
            try:
                return _YOLOPose(str(mp), device=device)
            except Exception:
                pass
    return _DummyPose()


# --------------------------- Keypoint extractor ---------------------------

class PoseEstimator:
    """
    关键点提取与归一化工具。
    返回形状 (17, 2) 的 ndarray（x,y），范围 [0,1]，若未检测到返回 None。
    """

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        arr = np.array(img.convert("RGB"))
        return arr[:, :, ::-1].copy()  # RGB -> BGR

    @staticmethod
    def extract_keypoints_pil(pil_img: Image.Image, detector: Any) -> Optional[np.ndarray]:
        if detector is None:
            return None

        img_bgr = PoseEstimator._pil_to_bgr(pil_img)
        try:
            preds = detector(img_bgr)
        except Exception:
            return None

        # ultralytics: list[Results]，每个 results.keypoints.data: (n,17,3)
        kp = None
        if preds and hasattr(preds[0], "keypoints") and preds[0].keypoints is not None:
            data = preds[0].keypoints.data  # tensor 或 ndarray
            try:
                import torch  # 延迟导入
                if "torch" in str(type(data)):  # 粗略判断是否为 torch.Tensor
                    data = data.detach().cpu().numpy()
            except Exception:
                # 没有 torch 也没关系，尽量转成 numpy
                try:
                    data = np.array(data)
                except Exception:
                    data = None

            if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[1] >= 17:
                # 选取平均置信度最高的一人
                conf_mean = data[:, :, 2].mean(axis=1)
                idx = int(conf_mean.argmax())
                xy = data[idx, :17, :2]  # (17,2)
                kp = xy

        if kp is None:
            return None

        # 归一化至 [0,1]
        h, w = img_bgr.shape[:2]
        if w <= 0 or h <= 0:
            return None
        kp_norm = kp.astype(np.float32)
        kp_norm[:, 0] /= float(w)
        kp_norm[:, 1] /= float(h)
        return kp_norm


# --------------------------- Skeleton visualization ---------------------------

# COCO 关键点连线对（简化常用）
_COCO_PAIRS: List[Tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # 手臂
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿
    (5, 6), (11, 12),                       # 肩与胯
    (0, 5), (0, 6), (5, 11), (6, 12)        # 躯干
]

def draw_skeleton(
    pil_img: Image.Image,
    keypoints: np.ndarray,
    radius: int = 4,
    line_width: int = 3,
    joint_fill: Tuple[int, int, int] = (255, 0, 0),
    line_fill: Tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """
    在图像上绘制骨架。keypoints 可以是像素坐标或 [0,1] 归一化坐标。
    """
    out = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    w, h = out.size
    pts = np.array(keypoints, dtype=float).copy()
    # 自动判定是否归一化
    if pts.size and pts.max() <= 1.0 + 1e-6:
        pts[:, 0] *= w
        pts[:, 1] *= h

    # 画连线
    for a, b in _COCO_PAIRS:
        if a < len(pts) and b < len(pts):
            ax, ay = pts[a]
            bx, by = pts[b]
            draw.line([(ax, ay), (bx, by)], width=line_width, fill=line_fill)

    # 画关键点
    for x, y in pts:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=joint_fill)

    return out


# 显式导出，确保外部可 from processors.pose import draw_skeleton
__all__ = ["make_pose_detector", "PoseEstimator", "draw_skeleton"]
