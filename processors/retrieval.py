# processors/retrieval.py
# ------------------------------------------------------------
# A retrieval helper with a unified 'Matcher' API.
# - Load FAISS index + ids + meta from disk
# - CLIP embedding search (open_clip)
# - Optional re-ranking with Pose and Color (graceful fallback)
# - Public-domain filter
# ------------------------------------------------------------

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
from PIL import Image

# ---------- Optional deps (OpenCV for color) ----------
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None  # type: ignore

# ---------- FAISS ----------
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("FAISS is required to use the Matcher. Please install faiss / faiss-cpu.") from e

# ---------- CLIP ----------
try:
    import torch
    import open_clip
except Exception as e:
    raise RuntimeError("open_clip & torch are required. Install them before using Matcher.") from e


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _safe_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    pp = Path(p)
    return str(pp) if pp.exists() else None


class Matcher:
    """
    Light-weight retrieval wrapper for app.py.

    You can initialize in two ways:

    1) Path-style (recommended for app.py):
        Matcher(
            faiss_index_path="indexing/faiss.index",
            ids_path="indexing/ids.npy",
            meta_path="indexing/meta.json",
            images_dir="data/images",
            device="mps|cuda|cpu"
        )

    2) Object-style (advanced):
        Matcher(index=<faiss.Index>, ids=<np.ndarray>, meta=<list[dict]>,
                model=<open_clip model>, preprocess=<transform>, device="cpu")

    Public APIs:
        - search_file(img_path, k=200, topn=6, weights={...}, filters={...})
        - search_pil(pil_image, ...)
        - search(...)  # alias of search_pil for backward compat
    """

    # ---------------------- INIT ----------------------
    def __init__(
        self,
        # object-style (optional)
        index: Any = None,
        ids: Optional[np.ndarray] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        model: Any = None,
        preprocess: Any = None,
        # path-style (recommended)
        faiss_index_path: Optional[str] = None,
        ids_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:

        self.device = device

        # --- Load index/ids/meta ---
        if index is not None and ids is not None and meta is not None:
            self.index = index
            self.ids = ids
            self.meta = meta
        else:
            if not (faiss_index_path and ids_path and meta_path):
                raise ValueError(
                    "Provide either (index, ids, meta) OR "
                    "(faiss_index_path, ids_path, meta_path)."
                )
            self.index = faiss.read_index(str(faiss_index_path))
            self.ids = np.load(str(ids_path))
            with open(str(meta_path), "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        # images root (used to resolve file_name -> full path)
        self.images_dir = Path(images_dir) if images_dir else None

        # --- Build id->meta map & ensure image_path exists ---
        self._id2meta: Dict[str, Dict[str, Any]] = {}
        for i, rid in enumerate(self.ids):
            m = dict(self.meta[i]) if i < len(self.meta) else {}
            # ensure image_path
            if not m.get("image_path"):
                fn = m.get("file_name")
                if self.images_dir and fn:
                    imgp = self.images_dir / fn
                    if imgp.exists():
                        m["image_path"] = str(imgp)
            self._id2meta[str(rid)] = m
            self.meta[i] = m  # keep normalized

        # --- Load CLIP (open_clip) if not provided ---
        if model is None or preprocess is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device
            )
        else:
            self.model = model
            self.preprocess = preprocess

        # --- Optional pose deps (graceful fallback) ---
        self._pose_ready = False
        self._pose_detector = None
        self._pose_extract_fn = None
        try:
            # We support either:
            #   - processors.pose.PoseEstimator.extract_keypoints_pil
            #   - processors.pose.extract_keypoints_pil
            from processors import pose as _pose_mod  # type: ignore
            if hasattr(_pose_mod, "make_pose_detector"):
                self._pose_detector = _pose_mod.make_pose_detector()
            # extractor function
            if hasattr(_pose_mod, "PoseEstimator"):
                self._pose_extract_fn = _pose_mod.PoseEstimator.extract_keypoints_pil
            elif hasattr(_pose_mod, "extract_keypoints_pil"):
                self._pose_extract_fn = _pose_mod.extract_keypoints_pil
            self._pose_ready = self._pose_detector is not None and self._pose_extract_fn is not None
        except Exception:
            self._pose_ready = False

        self._warned_pose = False
        self._warned_color = False

    # ---------------------- Embedding ----------------------
    def _clip_embed_pil(self, img: Image.Image) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype("float32")  # (1, d)

    # ---------------------- Auxiliary scores ----------------------
    def _color_hist_score(self, a_img: Image.Image, b_img_path: Optional[str]) -> float:
        if not _HAS_CV2:
            if not self._warned_color:
                warnings.warn("[Matcher] OpenCV not found; color re-ranking is disabled.")
                self._warned_color = True
            return 0.0
        try:
            if not b_img_path:
                return 0.0
            # query: PIL -> BGR
            a = np.array(a_img)[:, :, ::-1]
            a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            a_hist = cv2.calcHist([a_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(a_hist, a_hist)

            b = cv2.imread(b_img_path)
            if b is None:
                return 0.0
            b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
            b_hist = cv2.calcHist([b_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(b_hist, b_hist)

            ah = a_hist.flatten().astype(np.float32)
            bh = b_hist.flatten().astype(np.float32)
            ah = _l2_normalize(ah)
            bh = _l2_normalize(bh)
            return float(np.dot(ah, bh))
        except Exception:
            return 0.0

    def _pose_score(self, a_img: Image.Image, b_img_path: Optional[str]) -> float:
        if not self._pose_ready:
            if not self._warned_pose:
                warnings.warn("[Matcher] Pose deps not found; pose re-ranking is disabled.")
                self._warned_pose = True
            return 0.0
        if not (b_img_path and Path(b_img_path).exists()):
            return 0.0
        try:
            # query
            a_kp = self._pose_extract_fn(a_img, self._pose_detector)  # type: ignore
            # candidate
            b_img = Image.open(b_img_path).convert("RGB")
            b_kp = self._pose_extract_fn(b_img, self._pose_detector)  # type: ignore
            if a_kp is None or b_kp is None:
                return 0.0
            a_vec = _l2_normalize(np.asarray(a_kp).flatten().astype(np.float32))
            b_vec = _l2_normalize(np.asarray(b_kp).flatten().astype(np.float32))
            return float(np.dot(a_vec, b_vec))
        except Exception:
            return 0.0

    # ---------------------- Core search ----------------------
    def _faiss_search(self, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (D, I) from faiss.search; shapes: (1, k)
        """
        return self.index.search(q_vec, k)

    def _combine_scores(
        self,
        clip_s: float,
        pose_s: float,
        color_s: float,
        w_clip: float,
        w_pose: float,
        w_color: float,
    ) -> float:
        w_sum = max(w_clip + w_pose + w_color, 1e-8)
        return float((w_clip * clip_s + w_pose * pose_s + w_color * color_s) / w_sum)

    def _postprocess_candidates(
        self,
        ids: Iterable[str],
        clip_scores: Iterable[float],
        q_img: Image.Image,
        weights: Dict[str, float],
        filters: Optional[Dict[str, Any]],
        topn: int,
    ) -> List[Dict[str, Any]]:
        require_pd = bool(filters.get("require_public_domain", False)) if filters else False

        items: List[Tuple[str, Dict[str, Any], float]] = []
        for rid, cs in zip(ids, clip_scores):
            m = self._id2meta.get(str(rid))
            if not m:
                continue
            if require_pd:
                lic = m.get("license") or ("Public Domain" if m.get("isPublicDomain") else "")
                if not lic:
                    continue

            b_path = m.get("image_path")
            if not b_path and m.get("file_name") and self.images_dir:
                cand = self.images_dir / m["file_name"]
                b_path = str(cand) if cand.exists() else None

            w_clip = float(weights.get("w_clip", 1.0))
            w_pose = float(weights.get("w_pose", 0.0))
            w_color = float(weights.get("w_color", 0.0))

            pose_s = self._pose_score(q_img, b_path) if w_pose > 0 else 0.0
            color_s = self._color_hist_score(q_img, b_path) if w_color > 0 else 0.0
            score = self._combine_scores(cs, pose_s, color_s, w_clip, w_pose, w_color)

            mm = dict(m)  # copy
            mm["score"] = float(score)
            items.append((str(rid), mm, float(score)))

        items.sort(key=lambda x: x[2], reverse=True)
        return [m for _, m, _ in items[:topn]]

    # ---------------------- Public APIs ----------------------
    def search_pil(
        self,
        q_img: Image.Image,
        k: int = 200,
        topn: int = 6,
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using a PIL image and return a list of result dicts:
        [{..., "score": float}, ...] sorted by score desc.
        """
        if weights is None:
            weights = {"w_clip": 1.0, "w_pose": 0.0, "w_color": 0.0}

        # 1) CLIP embedding -> FAISS
        q_vec = self._clip_embed_pil(q_img)               # (1, d)
        D, I = self._faiss_search(q_vec, k=k)             # (1, k)
        d = D.reshape(-1)

        # normalize FAISS distances/sim to [0,1] as "clip score"
        if len(d) > 0:
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                clip_scores = (d - d_min) / (d_max - d_min)
            else:
                clip_scores = np.zeros_like(d)
        else:
            clip_scores = np.zeros_like(d)

        ids = [str(self.ids[idx]) for idx in I.reshape(-1)]

        # 2) Postprocess: filter + (pose/color) re-ranking
        return self._postprocess_candidates(
            ids=ids,
            clip_scores=clip_scores,
            q_img=q_img.convert("RGB"),
            weights=weights,
            filters=filters,
            topn=topn,
        )

    # Backward compatibility alias
    def search(
        self,
        q_img: Image.Image,
        k: int = 200,
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        topn: int = 6,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Return legacy-style tuples (rid, meta, score).
        """
        results = self.search_pil(q_img, k=k, topn=topn, weights=weights, filters=filters)
        out: List[Tuple[str, Dict[str, Any], float]] = []
        for r in results:
            # find rid from id field if present, otherwise try reverse-lookup
            rid = str(r.get("id", ""))
            if not rid:
                # slow fallback: find the first id whose meta matches the dict reference
                # (only used for legacy consumers; app.py doesn't need it)
                for key, mm in self._id2meta.items():
                    if mm is r:
                        rid = key
                        break
            out.append((rid, r, float(r.get("score", 0.0))))
        return out

    def search_file(
        self,
        img_path: str,
        k: int = 200,
        topn: int = 6,
        weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper used by app.py — search from an image path.
        """
        p = Path(img_path)
        if not p.exists():
            raise FileNotFoundError(f"query image not found: {img_path}")
        pil = Image.open(str(p)).convert("RGB")
        return self.search_pil(pil, k=k, topn=topn, weights=weights, filters=filters)


__all__ = ["Matcher"]
