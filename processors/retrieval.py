# processors/retrieval.py
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Optional

import os
import json
import cv2
import numpy as np
from PIL import Image

from . import pose as pose_mod
from .similarity import pose_similarity, color_similarity, combine_score


_DEFAULT_WEIGHTS = dict(w_clip=0.6, w_pose=0.3, w_color=0.1)


class Matcher:
    """
    Retrieval wrapper: FAISS shortlist by CLIP, then re-rank with pose & color.
    The index order must align with `ids` and `meta` arrays (same length).

    Optionally, a candidate-pose cache can be provided (np.ndarray of length N,
    where each item is (K,2) keypoints in [0,1] or None). If not provided,
    the ctor will try to auto-load 'indexing/pose_kps.npy' when present.
    """

    def __init__(
        self,
        faiss_index: Any,
        id_list: Sequence[Any],
        meta_list: Sequence[Dict[str, Any]],
        clip_model: Any,
        clip_preprocess: Any,
        device: str = "cpu",
        pose_cache: Optional[np.ndarray] = None,
    ):
        self.index = faiss_index
        self.ids = list(id_list)
        self.meta = list(meta_list)
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device

        # Optional: load pose cache if not passed in
        self.pose_cache = pose_cache
        if self.pose_cache is None:
            try:
                cache_path = os.path.join("indexing", "pose_kps.npy")
                if os.path.exists(cache_path):
                    pc = np.load(cache_path, allow_pickle=True)
                    if len(pc) == len(self.meta):
                        self.pose_cache = pc
                    else:
                        # length mismatch -> ignore
                        self.pose_cache = None
            except Exception:
                self.pose_cache = None

    # ----------------------------
    # Encoders
    # ----------------------------
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        """Return L2-normalized CLIP embedding as (1, D) float32."""
        import torch
        with torch.no_grad():
            t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            return f.cpu().numpy().astype("float32")

    # ----------------------------
    # Search
    # ----------------------------
    def search(
        self,
        pil_img: Image.Image,
        k: int = 200,
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        topn: int = 50,
    ) -> List[Tuple[Any, Dict[str, Any], float]]:
        """
        Args:
            pil_img: query PIL image.
            k: FAISS shortlist size (before re-ranking).
            weights: dict with keys {'w_clip','w_pose','w_color'}.
            filters: e.g. {'require_public_domain': True}.
            topn: number of final results to return.

        Returns:
            List of tuples: (artwork_id, meta_dict, fused_score)
        """
        if k <= 0:
            return []
        wts = _DEFAULT_WEIGHTS | (weights or {})
        require_pd = bool(filters.get("require_public_domain", False)) if filters else False

        # 1) shortlist by CLIP+FAISS
        q = self.encode_image(pil_img)
        sims, idxs = self.index.search(q, int(k))
        idxs = idxs[0]
        base = [(int(i), float(s)) for i, s in zip(idxs, sims[0]) if i >= 0]

        if not base:
            return []

        # 2) query auxiliaries
        pr = pose_mod.extract_pose(pil_img)
        q_kps = pr.keypoints
        q_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 3) re-rank with pose & color (+ small public-domain bonus)
        ranked: List[Tuple[int, float]] = []
        for i, s_clip in base:
            # defensive bound check (in case of stale index)
            if i < 0 or i >= len(self.meta):
                continue
            m = self.meta[i]

            # optional license filter
            if require_pd:
                lic = str(m.get("license", "")).lower()
                if lic not in ("public domain", "cc0"):
                    continue

            cand_bgr = None
            img_path = m.get("image_path")
            if img_path and os.path.exists(img_path):
                cand_bgr = cv2.imread(img_path)

            # pose
            cand_kps = None
            if self.pose_cache is not None and i < len(self.pose_cache):
                cand_kps = self.pose_cache[i]
            sim_pose = pose_similarity(q_kps, cand_kps) if (q_kps is not None and cand_kps is not None) else 0.0

            # color
            sim_color = color_similarity(q_bgr, cand_bgr) if cand_bgr is not None else 0.0

            # tiny meta bonus for public-domain/CC0
            bonus = 0.02 if str(m.get("license", "")).lower() in ("public domain", "cc0") else 0.0

            score = combine_score(s_clip, sim_pose, sim_color, meta_bonus=bonus, w=wts)
            ranked.append((i, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[: max(1, int(topn))]
        return [(self.ids[i], self.meta[i], sc) for i, sc in top]
