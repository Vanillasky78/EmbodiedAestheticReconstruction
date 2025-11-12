# -*- coding: utf-8 -*-
"""
Pose / Image matcher for EAR.

Public methods:
  - match_pil(pil, topk=3)
  - match_image_bytes(b, topk=3)
  - match_image_embedding(q, topk=3)

Returns list of dicts: { filename, title, artist, year, score }
"""

import io
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import open_clip

from ..config import Settings


def _detect_device(pref: Optional[str] = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _nan_to_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


class PoseMatcher:
    def __init__(self, settings: Settings, museum: str = "local"):
        self.settings = settings
        self.museum = museum

        self.data_dir = os.path.join(self.settings.data_root, museum)
        embed_path = os.path.join(self.data_dir, "embeddings.npy")
        meta_path = os.path.join(self.data_dir, "embeddings_meta.csv")

        if not os.path.exists(embed_path):
            raise FileNotFoundError(f"Embeddings not found: {embed_path}")

        self.embeddings = np.load(embed_path).astype("float32")  # (N, D)
        self.embeddings = _l2n(self.embeddings)

        # load meta (BOM-safe), require filename
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path, encoding="utf-8-sig")
            df.columns = [c.strip().lower() for c in df.columns]
        else:
            df = pd.DataFrame({"filename": []})

        if "filename" not in df.columns:
            raise ValueError(f"'filename' column is required in {meta_path}")

        self.meta_df = df
        self.filenames = df["filename"].map(lambda x: os.path.basename(str(x))).tolist()

        if len(self.filenames) != self.embeddings.shape[0]:
            # 长度不一致不致命，但提醒修正
            print("[WARN] embeddings and meta length mismatch; check your build step.")

        # CLIP model for query encoding (when using match_pil / match_image_bytes)
        self.device = _detect_device(self.settings.device_override)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.settings.clip_model_name, pretrained=self.settings.clip_pretrained, device=self.device
        )
        self.clip_model.eval()

    # ---------- helpers ----------

    def _row_to_payload(self, idx: int, score: float) -> Dict[str, Any]:
        d: Dict[str, Any] = {"filename": None, "title": None, "artist": None, "year": None, "score": float(score)}
        if 0 <= idx < len(self.meta_df):
            row = {k: _nan_to_none(v) for k, v in self.meta_df.iloc[idx].to_dict().items()}
            d["filename"] = row.get("filename")

            # 常见字段
            d["title"] = row.get("title")
            d["artist"] = row.get("artist")

            y = row.get("year")
            if y is not None:
                try:
                    d["year"] = int(float(y))
                except Exception:
                    d["year"] = None
        return d

    def _topk_from_query_vec(self, q: np.ndarray, topk: int) -> List[Dict[str, Any]]:
        # cosine sim (embeddings already L2 normalized)
        sims = (self.embeddings @ q.reshape(-1)).astype("float32")
        idx = np.argsort(-sims)[: int(topk)]
        return [self._row_to_payload(int(i), float(sims[i])) for i in idx]

    # ---------- public APIs ----------

    def match_image_embedding(self, q: np.ndarray, topk: int = 3) -> List[Dict[str, Any]]:
        """q: (D,) numpy vector already L2-normalized"""
        if q.ndim != 1:
            q = q.reshape(-1)
        q = q.astype("float32")
        q = _l2n(q)
        return self._topk_from_query_vec(q, topk)

    def _encode_pil_dual(self, pil: Image.Image) -> np.ndarray:
        """encode a PIL to a single (D,) vector using dual-view average."""
        # full
        im_full = self.preprocess(pil.convert("RGB"))
        # square center crop
        w, h = pil.size
        m = min(w, h)
        pil_square = pil.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
        im_square = self.preprocess(pil_square.convert("RGB"))

        ims = torch.stack([im_full, im_square], dim=0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=str(self.device), enabled=(self.device.type in ["cuda", "mps"])):
            feat = self.clip_model.encode_image(ims)  # (2, D)
            feat = feat.float().cpu().numpy().mean(axis=0)  # (D,)
        return _l2n(feat)

    def match_pil(self, pil: Image.Image, topk: int = 3) -> List[Dict[str, Any]]:
        q = self._encode_pil_dual(pil)
        return self._topk_from_query_vec(q, topk)

    def match_image_bytes(self, b: bytes, topk: int = 3) -> List[Dict[str, Any]]:
        pil = Image.open(io.BytesIO(b))
        return self.match_pil(pil, topk=topk)
