# processors/retrieval.py
# ------------------------------------------------------------
# A retrieval helper with a unified 'Matcher' API.
# - FAISS candidate retrieval by CLIP embedding
# - Optional re-ranking with Pose and Color
# - Backward-compatible: accept either in-memory objects OR file paths

from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import cv2  # optional: color histogram
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None  # type: ignore

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)

def _maybe_load_faiss(index_path: Optional[str]):
    if not index_path:
        return None
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS is required to load an index from file, but it's not available.") from e
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(index_path)

def _maybe_autoload_clip(device: str):
    """
    Autoload a small OpenCLIP if model/preprocess are not provided.
    Uses ViT-B-32 (openai) to ensure portability on CPU/MPS.
    """
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.to(device)
    model.eval()
    return model, preprocess

class Matcher:
    """
    A light wrapper used by app.py & match_pose_to_artworks.py

    You can init it in TWO ways (both等价):

    1) 传对象（原始用法，与你当前retrieval.py一致）
       Matcher(index=faiss_index, ids=np_array, meta=list_of_dict, model=clip_model, preprocess=clip_preprocess, device="mps")

    2) 传路径（与 app.py 的“faiss_index_path / ids_path / meta_path”参数名兼容）
       Matcher(
         faiss_index_path="indexing/faiss.index",
         ids_path="indexing/ids.npy",
         meta_path="indexing/meta.json",
         images_dir="data/images",       # 用来补全 meta['image_path']
         device="mps",                   # 可选；默认自动装载 OpenCLIP
         autoload_model=True             # 不提供model/preprocess时，会自动加载 ViT-B-32(openai)
       )
    """

    def __init__(
        self,
        # --- 原始对象式 ---
        index: Any = None,
        ids: Optional[np.ndarray] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        model: Any = None,
        preprocess: Any = None,

        # --- 路径式（向后兼容 app.py 中的关键字）---
        faiss_index_path: Optional[str] = None,
        index_path: Optional[str] = None,      # 兼容别的命名
        index_file: Optional[str] = None,      # 兼容别的命名
        ids_path: Optional[str] = None,
        ids_file: Optional[str] = None,        # 兼容别的命名
        meta_path: Optional[str] = None,
        meta_file: Optional[str] = None,       # 兼容别的命名
        images_dir: Optional[str] = None,

        # --- 其它 ---
        device: str = "cpu",
        autoload_model: bool = True,
    ):
        # ---- 选择性从路径加载 ----
        # 兼容不同关键字：faiss_index_path / index_path / index_file
        index_path_any = faiss_index_path or index_path or index_file
        ids_path_any   = ids_path or ids_file
        meta_path_any  = meta_path or meta_file

        if index is None and (index_path_any is not None):
            index = _maybe_load_faiss(index_path_any)

        if ids is None and ids_path_any:
            if not Path(ids_path_any).exists():
                raise FileNotFoundError(f"ids not found: {ids_path_any}")
            ids = np.load(ids_path_any)

        if meta is None and meta_path_any:
            if not Path(meta_path_any).exists():
                raise FileNotFoundError(f"meta not found: {meta_path_any}")
            with open(meta_path_any, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # 补全 image_path
            if images_dir:
                for m in meta:
                    # 若已有 image_path 则保留；否则用 file_name 拼出完整路径
                    if not m.get("image_path"):
                        fn = m.get("file_name") or m.get("filename")
                        if fn:
                            m["image_path"] = str(Path(images_dir) / fn)

        # ---- 基本校验 ----
        if index is None or ids is None or meta is None:
            raise ValueError(
                "[Matcher] Missing core components. "
                "Provide (index, ids, meta) or the corresponding file paths."
            )

        # 若未提供模型与预处理，则按需自动加载
        if model is None or preprocess is None:
            if autoload_model:
                model, preprocess = _maybe_autoload_clip(device)
            else:
                raise ValueError(
                    "[Matcher] model/preprocess not provided and autoload_model=False."
                )

        self.index = index
        self.ids = ids
        self.meta = meta
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # id -> meta 映射，便于查元数据
        self._id2meta: Dict[str, Dict[str, Any]] = {}
        # ids 可能是整型数组或字符串；统一为字符串索引
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
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
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
        D, I = self.index.search(q, k)   # distances -> normalize into similarity
        d = D.reshape(-1)

        # 归一：0~1，相似度越大越好（不同FAISS度量差异在此统一）
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

        # 3) 计算 pose/color 分数
        final_items: List[Tuple[str, Dict[str, Any], float]] = []
        for rid, m, clip_s in cand:
            img_path = m.get("image_path")
            pose_s = self._pose_score(q_img, img_path) if w_pose > 0 else 0.0
            color_s = self._color_hist_score(q_img, img_path) if w_color > 0 else 0.0
            score = (w_clip * clip_s + w_pose * pose_s + w_color * color_s) / w_sum
            final_items.append((rid, m, float(score)))

        final_items.sort(key=lambda x: x[2], reverse=True)
        return final_items[:topn]

__all__ = ["Matcher"]
