#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Pose + CLIP Matcher with Value / Portrait / Masterpiece Priors
---------------------------------------------------------------------

Data layout (per museum):
    data/<museum>/
        embeddings.npy          # (N, D_clip)  L2-normalized or raw
        pose_embeddings.npy     # (N, D_pose)  L2-normalized or raw (optional)
        embeddings_meta.csv     # filename + metadata

Scoring:

    base_sim   = (1 - pose_weight) * clip_sim + pose_weight * pose_sim

    bonus      = alpha_value      * value_score
               + beta_portrait    * portrait_flag
               + gamma_master     * masterpiece_flag
               + delta_tier       * tier_score

    final_sim  = base_sim * (1 + bonus)

Filtering modes (optional):

    mode="default"               → no hard filter, only soft bonus
    mode="portrait_only"         → keep portrait_flag == 1
    mode="high_value_only"       → keep value_score >= high_value_threshold
    mode="portrait_high_value"   → both portrait & high value
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import normalize_rows  # already defined in utils.py


class PoseMatcher:
    """
    Hybrid matcher combining CLIP embeddings + Pose embeddings + metadata priors.
    """

    def __init__(
        self,
        museum_dir: Path,
        pose_weight: float = 0.35,
        topk_default: int = 3,
        alpha_value: float = 0.40,          # weight for value_score
        beta_portrait: float = 0.35,        # weight for portrait_flag
        gamma_master: float = 0.25,         # weight for masterpiece_flag
        delta_tier: float = 0.15,           # weight for tier_score
        high_value_threshold: float = 0.60, # threshold for "high value"
    ):
        self.museum_dir = museum_dir
        self.pose_weight = float(pose_weight)
        self.topk_default = int(topk_default)

        self.alpha_value = float(alpha_value)
        self.beta_portrait = float(beta_portrait)
        self.gamma_master = float(gamma_master)
        self.delta_tier = float(delta_tier)
        self.high_value_threshold = float(high_value_threshold)

        emb_path = museum_dir / "embeddings.npy"
        meta_path = museum_dir / "embeddings_meta.csv"
        pose_path = museum_dir / "pose_embeddings.npy"

        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings: {emb_path}")

        # ---- load CLIP embeddings ----
        self.clip_emb = np.load(emb_path).astype("float32")
        self.clip_emb = normalize_rows(self.clip_emb)

        # ---- load metadata ----
        self.meta: List[Dict] = self._load_meta_rows(meta_path)

        # ---- load pose embeddings (optional) ----
        if pose_path.exists():
            print(f"[INFO] Pose embeddings found at {pose_path}")
            self.pose_emb = np.load(pose_path).astype("float32")
            self.pose_emb = normalize_rows(self.pose_emb)
            self.has_pose = True
        else:
            print(f"[INFO] pose_embeddings.npy not found → CLIP-only mode")
            self.pose_emb = None
            self.has_pose = False

        # ---- sanity checks ----
        if len(self.meta) != len(self.clip_emb):
            raise ValueError(
                f"Metadata length {len(self.meta)} != "
                f"CLIP embeddings length {len(self.clip_emb)}"
            )
        if self.has_pose and len(self.pose_emb) != len(self.clip_emb):
            raise ValueError(
                f"Pose embeddings length {len(self.pose_emb)} != "
                f"CLIP embeddings length {len(self.clip_emb)}"
            )

        # ---- build metadata priors as numpy arrays ----
        (
            self.value_scores,
            self.portrait_flags,
            self.masterpiece_flags,
            self.tier_scores,
        ) = self._build_meta_arrays(self.meta)

        print(
            f"[OK] Loaded museum='{museum_dir.name}' "
            f"→ N={len(self.clip_emb)}, pose={self.has_pose}"
        )

    # ------------------------------------------------------------------
    # Loading / preprocessing
    # ------------------------------------------------------------------

    def _load_meta_rows(self, path: Path) -> List[Dict]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")

        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def _safe_float(row: Dict, *keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except Exception:
                    continue
        return float(default)

    @staticmethod
    def _safe_flag(row: Dict, *keys: str, default: int = 0) -> int:
        for k in keys:
            if k in row and row[k] not in ("", None):
                v = str(row[k]).strip().lower()
                if v in ("1", "true", "yes", "y"):
                    return 1
                if v in ("0", "false", "no", "n"):
                    return 0
        return int(default)

    def _build_meta_arrays(
        self, rows: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build dense numpy arrays for metadata priors:
          value_scores      ∈ [0,1]
          portrait_flags    ∈ {0,1}
          masterpiece_flags ∈ {0,1}
          tier_scores       ∈ [0,1]
        """

        value_scores: List[float] = []
        portrait_flags: List[float] = []
        masterpiece_flags: List[float] = []
        tier_scores: List[float] = []

        for r in rows:
            # value_score or fallback to some alternative field
            vs = self._safe_float(r, "value_score", "value_norm", default=0.0)
            vs = max(0.0, min(1.0, vs))

            pf = self._safe_flag(r, "portrait_flag", "is_portrait", default=0)

            mf = self._safe_flag(
                r, "masterpiece_flag", "is_masterpiece", "highlight", default=0
            )

            # tier_score (0..1) or tier (1,2,3)
            ts = self._safe_float(r, "tier_score", default=-1.0)
            if ts < 0.0:  # not provided, fall back to 'tier'
                tier_raw = self._safe_float(r, "tier", default=0.0)
                # simple mapping: 1 → 1.0, 2 → 0.7, 3 → 0.4, else 0.5
                if tier_raw == 1:
                    ts = 1.0
                elif tier_raw == 2:
                    ts = 0.7
                elif tier_raw == 3:
                    ts = 0.4
                else:
                    ts = 0.5
            ts = max(0.0, min(1.0, ts))

            value_scores.append(vs)
            portrait_flags.append(pf)
            masterpiece_flags.append(mf)
            tier_scores.append(ts)

        return (
            np.array(value_scores, dtype="float32"),
            np.array(portrait_flags, dtype="float32"),
            np.array(masterpiece_flags, dtype="float32"),
            np.array(tier_scores, dtype="float32"),
        )

    # ------------------------------------------------------------------
    # Query encoders (you can keep using your existing CLIP / pose code)
    # ------------------------------------------------------------------

    def encode_query_image(self, pil, clip_model, preprocess, device):
        """
        Encode query image using CLIP image encoder.
        Returns a 1 x D_clip L2-normalized numpy array.
        """
        import torch

        x = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=str(device), enabled=(device.type in ["cuda", "mps"])
        ):
            feat = clip_model.encode_image(x)
        feat = feat.float().cpu().numpy()
        feat = normalize_rows(feat)
        return feat  # (1, D)

    def compute_pose_for_query(
        self, pose_vec_1d: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Receives a 1D pose vector (same encoding as pose_embeddings.npy).
        Normalizes it and returns 1 x D_pose. If None or no pose index
        is available, returns None.
        """
        if pose_vec_1d is None or not self.has_pose:
            return None
        v = pose_vec_1d.astype("float32")[None, :]
        v = normalize_rows(v)
        return v  # (1, D_pose)

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match(
        self,
        clip_query_vec: np.ndarray,                  # shape (1, D_clip)
        pose_query_vec: Optional[np.ndarray] = None, # shape (1, D_pose) or None
        topk: Optional[int] = None,
        mode: str = "default",                       # "default", "portrait_only", ...
    ) -> List[Dict]:
        """
        Return top-k matches with hybrid similarity & metadata priors.

        Parameters
        ----------
        clip_query_vec : np.ndarray
            L2-normalized CLIP query vector, shape (1, D_clip).

        pose_query_vec : np.ndarray or None
            L2-normalized pose query vector, shape (1, D_pose).
            If None or pose embeddings are not loaded, the matcher
            falls back to CLIP-only similarity.

        topk : int
            Number of results to return. Defaults to self.topk_default.

        mode : str
            Selects filtering strategy:
              - "default"             → no hard filters, only soft priors
              - "portrait_only"       → keep portrait_flag == 1
              - "high_value_only"     → keep value_score >= high_value_threshold
              - "portrait_high_value" → both portrait & high value
        """
        if topk is None:
            topk = self.topk_default

        # -------- 1) CLIP similarity
        # clip_emb: (N, D_clip), clip_query_vec: (1, D_clip)
        sim_clip = (self.clip_emb @ clip_query_vec.T).reshape(-1)  # (N,)

        # -------- 2) Pose similarity (if available)
        if self.has_pose and pose_query_vec is not None:
            sim_pose = (self.pose_emb @ pose_query_vec.T).reshape(-1)  # (N,)
        else:
            sim_pose = None

        # -------- 3) Hybrid base similarity (pose + CLIP)
        if sim_pose is not None:
            w = float(self.pose_weight)
            base_sim = (1.0 - w) * sim_clip + w * sim_pose
        else:
            base_sim = sim_clip

        # -------- 4) Optional hard filters based on mode
        mask = np.ones_like(base_sim, dtype=bool)

        if mode in ("portrait_only", "portrait_high_value"):
            mask &= self.portrait_flags >= 0.5

        if mode in ("high_value_only", "portrait_high_value"):
            mask &= self.value_scores >= self.high_value_threshold

        # If mask is too restrictive and empties the candidate set,
        # fall back gracefully to "default" (no hard filter).
        if not np.any(mask):
            mask = np.ones_like(base_sim, dtype=bool)

        # Apply mask
        candidate_indices = np.where(mask)[0]
        base_sim = base_sim[candidate_indices]

        value_scores = self.value_scores[candidate_indices]
        portrait_flags = self.portrait_flags[candidate_indices]
        masterpiece_flags = self.masterpiece_flags[candidate_indices]
        tier_scores = self.tier_scores[candidate_indices]

        if sim_pose is not None:
            sim_pose = sim_pose[candidate_indices]
        sim_clip_sub = sim_clip[candidate_indices]

        # -------- 5) Metadata bonus
        bonus = (
            self.alpha_value * value_scores
            + self.beta_portrait * portrait_flags
            + self.gamma_master * masterpiece_flags
            + self.delta_tier * tier_scores
        )

        # Clip bonus to avoid exploding values
        bonus = np.clip(bonus, 0.0, 2.0)

        final_sim = base_sim * (1.0 + bonus)

        # -------- 6) Sort within candidate set
        order = np.argsort(final_sim)[::-1][:topk]
        chosen_indices = candidate_indices[order]

        # -------- 7) Pack results
        out: List[Dict] = []
        for idx_ranked, idx_global in enumerate(chosen_indices):
            row = self.meta[idx_global]

            filename = (
                row.get("filename")
                or row.get("file")
                or row.get("image_path")
                or row.get("path")
            )

            result = dict(
                rank=idx_ranked + 1,
                index=int(idx_global),
                filename=filename,
                score=float(final_sim[order[idx_ranked]]),
                clip=float(sim_clip_sub[order[idx_ranked]]),
                pose=float(sim_pose[order[idx_ranked]]) if sim_pose is not None else None,
                value_score=float(self.value_scores[idx_global]),
                portrait_flag=int(self.portrait_flags[idx_global]),
                masterpiece_flag=int(self.masterpiece_flags[idx_global]),
                tier_score=float(self.tier_scores[idx_global]),
                meta=row,
            )
            out.append(result)

        return out
