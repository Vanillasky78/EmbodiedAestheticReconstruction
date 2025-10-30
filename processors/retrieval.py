# processors/retrieval.py
# ------------------------------------------------------------
# A simple retrieval helper with a unified 'Matcher' API.
# - FAISS candidate retrieval by CLIP embedding
# - Optional re-ranking with Pose and Color
# - Graceful fallback if pose/color deps are not available

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # for color histogram (optional)
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None  # type: ignore


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


class Matcher:
    """
    A light wrapper used by app.py & match_pose_to_artworks.py

    Parameters
    ----------
    index : faiss.Index
        FAISS index built on CLIP image embeddings.
    ids : np.ndarray
        Array of record ids aligned with index rows.
    meta : List[dict]
        Metadata list aligned with ids (same length).
    model : open_clip model
        CLIP model (image encoder used).
    preprocess : torchvision-like transform
        Preprocess function to turn PIL into CLIP input.
    device : str
        "cuda" / "mps" / "cpu"
    """

    def __init__(
        self,
        index: Any,
        ids: np.ndarray,
        meta: List[Dict[str, Any]],
        model: Any,
        preprocess: Any,
        device: str = "cpu",
    ):
        self.index = index
        self.ids = ids
        self.meta = meta
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # id -> meta 映射，便于查元数据
        self._id2meta: Dict[str, Dict[str, Any]] = {}
        for i, rid in enumerate(self.ids):
            self._id2meta[str(rid)] = self.meta[i]

        # 姿态依赖：可用则加载，不可用就降级
        self._pose_ready = False
        try:
            from processors.pose import make_pose_detector, PoseEstimator  # type: ignore
            self._pose_detector = make_pose_detector()
            self._PoseEstimator = PoseEstimator
            self._pose_ready = True
        except Exception:
            self._pose_ready = False

        self._warned_pose = False
        self._warned_color = False

    # ---------- Embeddings ----------
    def _clip_embed_pil(self, img: Image.Image) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype("float32")  # (1, d)

    # ---------- Scores ----------
    def _color_hist_score(self, a_img: Image.Image, b_img_path: str) -> float:
        if not _HAS_CV2:
            if not self._warned_color:
                warnings.warn("[Matcher] OpenCV not found; color re-ranking is disabled.")
                self._warned_color = True
            return 0.0
        try:
            # query: PIL -> BGR
            a = np.array(a_img)[:, :, ::-1]
            a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            a_hist = cv2.calcHist([a_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(a_hist, a_hist)

            # candidate
            if not b_img_path or not Path(b_img_path).exists():
                return 0.0
            b = cv2.imread(b_img_path)
            if b is None:
                return 0.0
            b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
            b_hist = cv2.calcHist([b_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(b_hist, b_hist)

            # 余弦
            ah = a_hist.flatten().astype(np.float32)
            bh = b_hist.flatten().astype(np.float32)
            ah = _l2_normalize(ah)
            bh = _l2_normalize(bh)
            return float(np.dot(ah, bh))
        except Exception:
            return 0.0

    def _pose_score(self, a_img: Image.Image, b_img_path: str) -> float:
        if not self._pose_ready:
            if not self._warned_pose:
                warnings.warn("[Matcher] Pose deps not found; pose re-ranking is disabled.")
                self._warned_pose = True
            return 0.0
        if not b_img_path or not Path(b_img_path).exists():
            return 0.0
        try:
            # 用内置姿态估计器快速提点并做余弦（简单版）
            a_kp = self._PoseEstimator.extract_keypoints_pil(a_img, self._pose_detector)
            b_img = Image.open(b_img_path).convert("RGB")
            b_kp = self._PoseEstimator.extract_keypoints_pil(b_img, self._pose_detector)
            if a_kp is None or b_kp is None:
                return 0.0
            a_vec = _l2_normalize(a_kp.flatten().astype(np.float32))
            b_vec = _l2_normalize(b_kp.flatten().astype(np.float32))
            return float(np.dot(a_vec, b_vec))
        except Exception:
            return 0.0

    # ---------- Search ----------
    def search(
        self,
        q_img: Image.Image,
        k: int = 200,
        weights: Dict[str, float] | None = None,
        filters: Dict[str, Any] | None = None,
        topn: int = 12,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Returns
        -------
        list of (rid, meta, score) sorted by score desc
        """
        if weights is None:
            weights = {"w_clip": 1.0, "w_pose": 0.0, "w_color": 0.0}
        w_clip  = float(weights.get("w_clip", 1.0))
        w_pose  = float(weights.get("w_pose", 0.0))
        w_color = float(weights.get("w_color", 0.0))
        w_sum = max(w_clip + w_pose + w_color, 1e-8)

        # 1) CLIP → FAISS 召回
        q = self._clip_embed_pil(q_img)  # (1,d)
        D, I = self.index.search(q, k)   # distances are inner-product or L2-based (built time)
        # FAISS 通常越小越好或越大越好取决于索引类型。这里把 D 线性归一后当成“相似度”
        d = D.reshape(-1)
        # 归一：转成 0~1，相似度越大越好
        if len(d) > 0:
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                clip_scores = (d - d_min) / (d_max - d_min)
            else:
                clip_scores = np.zeros_like(d)
        else:
            clip_scores = np.zeros_like(d)

        ids = [str(self.ids[idx]) for idx in I.reshape(-1)]

        # 2) 过滤（如 Public Domain）
        require_pd = bool(filters.get("require_public_domain", False)) if filters else False
        cand: List[Tuple[str, Dict[str, Any], float]] = []
        for rid, cs in zip(ids, clip_scores):
            m = self._id2meta.get(rid)
            if m is None:
                continue
            if require_pd:
                lic = m.get("license") or ("Public Domain" if m.get("isPublicDomain") else "")
                if not lic:
                    continue
            cand.append((rid, m, float(cs)))

        if not cand:
            return []

        # 3) 计算 pose/color 分数（只对候选做，避免太慢）
        final_items: List[Tuple[str, Dict[str, Any], float]] = []
        for rid, m, clip_s in cand:
            pose_s = self._pose_score(q_img, m.get("image_path")) if w_pose > 0 else 0.0
            color_s = self._color_hist_score(q_img, m.get("image_path")) if w_color > 0 else 0.0
            # 归一化加权
            score = (w_clip * clip_s + w_pose * pose_s + w_color * color_s) / w_sum
            final_items.append((rid, m, float(score)))

        # 4) 排序 & 取前 topn
        final_items.sort(key=lambda x: x[2], reverse=True)
        return final_items[:topn]


__all__ = ["Matcher"]
