# backend/model/pose_matcher.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ✅ 相对导入（关键改动）
from ..config import Settings

try:
    import open_clip
except Exception as e:
    raise RuntimeError(
        "open_clip_torch is required. pip install open_clip_torch"
    ) from e


def _pick_device(override: Optional[str] = None) -> torch.device:
    """Choose best available device."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon Metal backend
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class PoseMatcher:
    """
    Loads CLIP model, museum embeddings, and provides matching API.

    Data layout expected:
      data/
        <museum_name>/
          images/                  # images (optional for runtime)
          embeddings.npy           # (N, D) float32
          embeddings_meta.csv      # CSV with 'filename' and optional 'title','artist','year'
    """

    def __init__(
        self,
        data_root: str = "data",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device_override: Optional[str] = None,
        use_dual_view: bool = True,
        verbose: bool = False,
    ):
        self.data_root = data_root
        self.device = _pick_device(device_override)
        self.device_str = str(self.device)
        self.use_dual_view = use_dual_view
        self.verbose = verbose

        # load CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.clip_model = model.eval().to(self.device)
        self.preprocess = preprocess

        # museums → dict of arrays/meta
        self.museums: Dict[str, Dict[str, np.ndarray | List[dict]]] = {}
        self._load_all_museums()

        # default museum = first one found (if exists)
        self.default_museum = next(iter(self.museums), None)

    # -------------------------------------------------

    @property
    def museums_loaded(self) -> List[str]:
        return list(self.museums.keys())

    @property
    def items_total(self) -> int:
        total = 0
        for m in self.museums.values():
            total += int(m["embeddings"].shape[0])
        return total

    # -------------------------------------------------

    def _load_all_museums(self) -> None:
        """Scan data_root for subfolders and load embeddings."""
        if not os.path.isdir(self.data_root):
            if self.verbose:
                print(f"[WARN] data_root not found: {self.data_root}")
            return

        for name in sorted(os.listdir(self.data_root)):
            folder = os.path.join(self.data_root, name)
            if not os.path.isdir(folder):
                continue
            emb_path = os.path.join(folder, "embeddings.npy")
            meta_csv = os.path.join(folder, "embeddings_meta.csv")
            if not os.path.isfile(emb_path) or not os.path.isfile(meta_csv):
                # silently skip incomplete museums
                if self.verbose:
                    print(f"[SKIP] {name}: missing embeddings.npy or embeddings_meta.csv")
                continue

            try:
                embs = np.load(emb_path).astype("float32")  # (N, D)
                meta = self._read_csv(meta_csv)  # list of dict rows
                if len(meta) != embs.shape[0]:
                    if self.verbose:
                        print(f"[WARN] {name}: meta rows != embeddings rows ({len(meta)} vs {embs.shape[0]})")
                # normalize (safety)
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
                self.museums[name] = {
                    "embeddings": embs,
                    "meta": meta,
                }
                if self.verbose:
                    print(f"[OK] loaded museum '{name}' → {embs.shape}")
            except Exception as e:
                if self.verbose:
                    print(f"[ERR] failed to load museum '{name}': {e}")

    @staticmethod
    def _read_csv(path: str) -> List[dict]:
        """Tiny CSV reader that returns list of dict (UTF-8)."""
        import csv

        rows: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k.strip(): v.strip() for k, v in r.items()})
        return rows

    # -------------------------------------------------

    @torch.inference_mode()
    def _embed_pil(self, img: Image.Image) -> torch.Tensor:
        """Compute L2-normalized CLIP embedding for a PIL image."""
        if not self.use_dual_view:
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.clip_model.encode_image(x)
            feat = _l2_normalize(feat.float()).cpu()
            return feat  # (1, D)

        # dual-view: full + center-crop (robustness to framing)
        full = self.preprocess(img).unsqueeze(0).to(self.device)

        # center crop square
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        crop = img.crop((left, top, left + side, top + side))
        cc = self.preprocess(crop).unsqueeze(0).to(self.device)

        f1 = self.clip_model.encode_image(full).float()
        f2 = self.clip_model.encode_image(cc).float()
        feat = _l2_normalize((f1 + f2) / 2.0).cpu()
        return feat  # (1, D)

    # -------------------------------------------------

    def match_pil(self, img: Image.Image, topk: int = 3, museum: Optional[str] = None) -> List[dict]:
        """Return top-k matches as list of dicts with meta + score."""
        if not self.museums:
            raise ValueError("No museums loaded. Please ensure embeddings exist under data/<museum>/.")

        use_museum = museum or self.default_museum
        if use_museum not in self.museums:
            raise ValueError(f"Museum '{use_museum}' not found. Loaded: {self.museums_loaded}")

        embs = self.museums[use_museum]["embeddings"]  # (N, D) float32
        meta = self.museums[use_museum]["meta"]

        q = self._embed_pil(img).numpy()[0]  # (D,)
        # cosine similarity = dot because all vectors L2-normalized
        sims = embs @ q  # (N,)

        k = int(min(topk, sims.shape[0]))
        idx = np.argpartition(-sims, k - 1)[:k]
        # sort within top-k
        idx = idx[np.argsort(-sims[idx])]

        out: List[dict] = []
        for i in idx:
            row = meta[i] if i < len(meta) else {}
            out.append(
                {
                    "filename": row.get("filename", f"item_{int(i)}"),
                    "title": row.get("title"),
                    "artist": row.get("artist"),
                    "year": row.get("year"),
                    "score": float(sims[i]),
                }
            )
        return out
