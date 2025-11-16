"""
PoseMatcher

CLIP-based matcher with optional *pose embeddings* re-weighting:

- loads CLIP embeddings from data/.../embeddings.npy
- loads optional pose embeddings from data/.../pose_embeddings.npy
- loads metadata from data/.../embeddings_meta.csv
- computes query CLIP (and pose) embeddings for the uploaded image
- mixes scores: (1 - pose_weight) * clip + pose_weight * pose
- returns Top-K matched artworks with their metadata
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
        # Must match build_embeddings.py
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device=self.device,
        )
        self.model.eval()

        # --------- Load CLIP embeddings ----------
        emb_path = settings.embeddings_path
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

        emb_np = np.load(emb_path)
        if isinstance(emb_np, np.lib.npyio.NpzFile):
            emb_np = emb_np["embeddings"]

        emb = torch.from_numpy(emb_np).float()
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.embeddings = emb.to(self.device)  # (N, D_clip)

        # --------- Load pose embeddings (optional) ----------
        self.pose_embeddings: Optional[torch.Tensor] = None
        self.pose_weight: float = float(settings.pose_weight)

        pose_path: Path = settings.pose_embeddings_path
        if pose_path.exists():
            pose_np = np.load(pose_path)
            # basic sanity check: same number of rows as CLIP embeddings
            if pose_np.shape[0] == emb_np.shape[0]:
                pose = torch.from_numpy(pose_np).float()
                pose = pose / (pose.norm(dim=-1, keepdim=True) + 1e-8)
                self.pose_embeddings = pose.to(self.device)  # (N, D_pose)
            else:
                print(
                    f"[WARN] pose_embeddings shape {pose_np.shape} "
                    f"does not match embeddings {emb_np.shape}; ignoring pose features."
                )
        else:
            # Not an error: we simply fall back to CLIP-only
            print(f"[INFO] pose_embeddings not found at {pose_path}, using CLIP-only scores.")

        # --------- Load metadata ----------
        meta_path = settings.embeddings_meta_path
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self.meta: List[Dict] = []
        self.meta_by_filename: Dict[str, Dict] = {}

        with meta_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
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
                    # keep all raw columns in case we need them later
                    **row,
                }
                self.meta.append(meta_entry)
                self.meta_by_filename[filename] = meta_entry

        # For compatibility with backend.main
        self.default_museum = settings.default_museum

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image into a normalized CLIP embedding (1, D_clip).
        """
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat  # (1, D_clip)

    def _encode_pose(self, image: Image.Image) -> torch.Tensor:
        """
        Encode pose-style features for the image.

        For now this is implemented as the same CLIP encoder
        (so that everything runs out-of-the-box). Later you can
        replace this with a dedicated pose encoder or a vector
        built from YOLO keypoints.

        Shape: (1, D_pose) — must match pose_embeddings' last dim.
        """
        # Placeholder implementation = CLIP features again
        return self._encode_image(image)

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
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

        # ---- CLIP similarity ----
        q_clip = self._encode_image(image)  # (1, D_clip)
        sims_clip = (self.embeddings @ q_clip.T).squeeze(1)  # (N,)

        # ---- pose similarity (optional) ----
        if self.pose_embeddings is not None and self.pose_weight > 0.0:
            q_pose = self._encode_pose(image)  # (1, D_pose)
            # if dims mismatch, fall back to CLIP-only
            if q_pose.shape[1] == self.pose_embeddings.shape[1]:
                sims_pose = (self.pose_embeddings @ q_pose.T).squeeze(1)
                alpha = float(self.pose_weight)
                sims = (1.0 - alpha) * sims_clip + alpha * sims_pose
            else:
                print(
                    f"[WARN] pose dim mismatch: query={q_pose.shape}, "
                    f"embeddings={self.pose_embeddings.shape}; using CLIP-only."
                )
                sims = sims_clip
        else:
            sims = sims_clip

        # Top-K indices
        k = min(topk, sims.shape[0])
        scores, indices = torch.topk(sims, k=k, largest=True, sorted=True)

        results: List[Dict] = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
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

    def match_image_bytes(
        self,
        image_bytes: bytes,
        museum: Optional[str] = None,
        topk: int = 3,
    ) -> List[Dict]:
        """
        Convenience wrapper used by the purely local Streamlit curatorial app.
        """
        from io import BytesIO

        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.match_pil(pil, museum=museum, topk=topk)
