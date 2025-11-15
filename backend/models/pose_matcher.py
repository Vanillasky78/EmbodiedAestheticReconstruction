"""
PoseMatcher

A lightweight CLIP-based matcher that:

- loads pre-computed embeddings from data/local/embeddings.npy
- loads metadata from data/local/embeddings_meta.csv
- computes an embedding for the query image
- returns the Top-K most similar artworks with their metadata

This version does NOT depend on YOLO pose directly; it only uses
the CLIP image encoder. You already created the embeddings with:

    python backend/tools/build_embeddings.py --museum local ...

So this matcher just consumes those files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image

from backend.config import Settings


@dataclass
class MatchItem:
    filename: str
    title: Optional[str]
    artist: Optional[str]
    year: Optional[int]
    score: float


class PoseMatcher:
    def __init__(self, settings: Settings):
        self.settings = settings

        # --------- Resolve device ----------
        if settings.device == "cpu":
            device = "cpu"
        elif settings.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif settings.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            # "auto" or unsupported → try CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        # --------- Load CLIP model ----------
        # It must match the model used in build_embeddings.py
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device=self.device,
        )
        self.model.eval()

        # --------- Load stored embeddings ----------
        emb_path = settings.embeddings_path
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

        # shape: (N, D)
        emb_np = np.load(emb_path)
        if isinstance(emb_np, np.lib.npyio.NpzFile):
            # Just in case someone saved np.savez
            # we expect a "embeddings" key
            emb_np = emb_np["embeddings"]

        # Store as normalized torch tensor on device
        emb = torch.from_numpy(emb_np).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.embeddings = emb.to(self.device)

        # --------- Load metadata ----------
        meta_path = settings.embeddings_meta_path
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self.meta: List[Dict] = []
        self.meta_by_filename: Dict[str, Dict] = {}

        with meta_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            # Simple CSV reader; no quoted commas expected here
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if not parts or not parts[0]:
                    continue
                row = dict(zip(header, parts))
                filename = row.get("filename") or row.get("file") or ""
                if not filename:
                    continue

                # Normalize year to int if possible
                raw_year = row.get("year") or row.get("Year")
                year_int: Optional[int] = None
                if raw_year:
                    try:
                        year_int = int(str(raw_year).split(",")[0].strip())
                    except Exception:
                        year_int = None

                meta_entry = {
                    "filename": filename,
                    "title": row.get("title") or row.get("Title"),
                    "artist": row.get("artist") or row.get("Artist"),
                    "year": year_int,
                    # Keep raw dict as well in case we need extra columns later
                    **row,
                }
                self.meta.append(meta_entry)
                self.meta_by_filename[filename] = meta_entry

        # For compatibility with the backend/main.py code
        self.default_museum = settings.default_museum

    # ------------------------------------------------------------------
    # Core API used both by backend and Streamlit frontend
    # ------------------------------------------------------------------
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a normalized CLIP embedding."""
        img = self.preprocess(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        with torch.no_grad():
            feat = self.model.encode_image(img)  # (1, D)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat  # (1, D)

    def match_pil(
        self,
        image: Image.Image,
        museum: Optional[str] = None,
        topk: int = 3,
    ) -> List[Dict]:
        """
        Match a PIL image against the stored embeddings.

        Returns a list of dicts:
            { "filename": ..., "title": ..., "artist": ..., "year": ..., "score": ... }
        """
        if topk <= 0:
            topk = 3

        # Encode query
        query = self._encode_image(image)  # (1, D)

        # Cosine similarity with all stored embeddings
        # Because both are normalized, dot-product == cosine
        sims = (self.embeddings @ query.T).squeeze(1)  # (N,)

        # Top-K indices
        k = min(topk, sims.shape[0])
        scores, indices = torch.topk(sims, k=k, largest=True, sorted=True)

        results: List[Dict] = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            # index into meta list; we assume same order as embeddings
            if 0 <= idx < len(self.meta):
                meta = self.meta[idx]
            else:
                meta = {}

            filename = meta.get("filename", "")
            title = meta.get("title")
            artist = meta.get("artist")
            year = meta.get("year")

            results.append(
                {
                    "filename": filename,
                    "title": title,
                    "artist": artist,
                    "year": year,
                    "score": float(score),
                }
            )

        return results