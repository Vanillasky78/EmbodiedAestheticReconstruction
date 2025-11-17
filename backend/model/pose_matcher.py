#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Pose+CLIP Matcher
------------------------
This module performs a hybrid similarity search over a museum dataset.

Data folder layout (per museum):
    data/<museum>/
        embeddings.npy          # (N, D_clip)
        pose_embeddings.npy     # (N, D_pose)   ← optional
        embeddings_meta.csv     # filenames + metadata

Final score:
    final = (1 - POSE_WEIGHT) * clip_sim + (POSE_WEIGHT) * pose_sim
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import normalize_rows, cosine_similarity_matrix


class PoseMatcher:
    """
    Hybrid matcher combining CLIP embeddings + Pose embeddings.
    """

    def __init__(
        self,
        museum_dir: Path,
        pose_weight: float = 0.35,        # You can tune this
        topk_default: int = 3,
    ):
        self.museum_dir = museum_dir
        self.pose_weight = float(pose_weight)
        self.topk_default = int(topk_default)

        # Load embeddings
        emb_path = museum_dir / "embeddings.npy"
        meta_path = museum_dir / "embeddings_meta.csv"
        pose_path = museum_dir / "pose_embeddings.npy"

        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings: {emb_path}")

        self.clip_emb = np.load(emb_path).astype("float32")
        self.clip_emb = normalize_rows(self.clip_emb)

        # Metadata
        self.meta = self._load_meta_rows(meta_path)

        # Pose embeddings (optional)
        if pose_path.exists():
            print(f"[INFO] Pose embeddings found at {pose_path}")
            self.pose_emb = np.load(pose_path).astype("float32")
            self.pose_emb = normalize_rows(self.pose_emb)
            self.has_pose = True
        else:
            print(f"[INFO] No pose_embeddings.npy → using CLIP-only mode")
            self.pose_emb = None
            self.has_pose = False

        # Basic sanity checks
        if len(self.meta) != len(self.clip_emb):
            raise ValueError(
                f"Metadata length {len(self.meta)} != embeddings length {len(self.clip_emb)}"
            )
        if self.has_pose and len(self.pose_emb) != len(self.clip_emb):
            raise ValueError(
                f"Pose embeddings length {len(self.pose_emb)} != CLIP embeddings length {len(self.clip_emb)}"
            )

        print(
            f"[OK] Loaded museum='{museum_dir.name}' "
            f"→ N={len(self.clip_emb)}, pose={self.has_pose}"
        )

    # -------------------------------------------------------------
    def _load_meta_rows(self, path: Path) -> List[Dict]:
        import csv

        if not path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    # -------------------------------------------------------------
    def encode_query_image(self, pil, clip_model, preprocess, device):
        """
        Encode query image using CLIP image encoder.
        """
        import torch

        x = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=str(device), enabled=(device.type in ["cuda", "mps"])):
            feat = clip_model.encode_image(x)
        feat = feat.float().cpu().numpy()
        feat = normalize_rows(feat)
        return feat  # (1, D)

    # -------------------------------------------------------------
    def compute_pose_for_query(self, kp_vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        You pass in a (D_pose,) vector computed from YOLO keypoints.
        This function simply normalizes it.
        """
        if kp_vec is None or not self.has_pose:
            return None
        v = kp_vec.astype("float32")[None, :]
        v = normalize_rows(v)
        return v  # (1, D_pose)

    # -------------------------------------------------------------
    def match(
        self,
        clip_query_vec: np.ndarray,
        pose_query_vec: Optional[np.ndarray] = None,
        topk: Optional[int] = None,
    ) -> List[Dict]:

        if topk is None:
            topk = self.topk_default

        # --- 1) CLIP similarity
        sim_clip = (self.clip_emb @ clip_query_vec.T).reshape(-1)

        # --- 2) Pose similarity (if enabled)
        if self.has_pose and pose_query_vec is not None:
            sim_pose = (self.pose_emb @ pose_query_vec.T).reshape(-1)
        else:
            sim_pose = None

        # --- 3) Hybrid scoring
        if sim_pose is not None:
            w = self.pose_weight
            final = (1.0 - w) * sim_clip + w * sim_pose
        else:
            final = sim_clip

        # --- 4) Sort
        idx = np.argsort(final)[::-1][:topk]

        # --- 5) Pack results
        out = []
        for i in idx:
            row = self.meta[i]
            out.append(
                dict(
                    filename=row.get("filename") or row.get("file"),
                    score=float(final[i]),
                    clip=float(sim_clip[i]),
                    pose=float(sim_pose[i]) if sim_pose is not None else None,
                    meta=row,
                )
            )
        return out
