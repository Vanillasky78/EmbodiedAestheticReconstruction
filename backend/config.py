#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global configuration for the EAR backend.

This file defines a simple Settings model that knows where your
data lives (embeddings, metadata CSV, images directory, pose
embeddings) and which device / weights to use for inference.
"""

from pathlib import Path
from pydantic import BaseModel


# Project root = repository root (EmbodiedAestheticReconstruction/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """
    Central configuration for the hybrid matcher.

    By default we point to the *mixed* global index, which already
    combines multiple sources (e.g. local + met + aic).
    """

    # Root folder that contains a single "museum" index:
    #   <data_root>/
    #       embeddings.npy
    #       embeddings_meta.csv
    #       pose_embeddings.npy (optional)
    #       images/
    data_root: Path = PROJECT_ROOT / "data" / "mixed"

    # Conceptual museum name used in API responses
    default_museum: str = "mixed"

    # Device: "auto", "cpu", "cuda", "mps"
    device: str = "auto"

    # ----- Hybrid similarity weights -----

    # Weight for pose embeddings when mixing scores:
    #   base_score = (1 - pose_weight) * clip_score + pose_weight * pose_score
    pose_weight: float = 0.35

    # Metadata priors:
    # final_score = base_score * (1 + bonus)
    # bonus      = alpha_value * value_score
    #            + beta_portrait * portrait_flag
    #            + gamma_master * masterpiece_flag
    #            + delta_tier * tier_score
    alpha_value: float = 0.40       # value / price prior
    beta_portrait: float = 0.35     # portrait preference
    gamma_master: float = 0.25      # masterpiece / highlight prior
    delta_tier: float = 0.15        # institution tier prior

    # Threshold for "high value" when using filtering modes
    high_value_threshold: float = 0.60

    class Config:
        arbitrary_types_allowed = True

    # ---------- paths for CLIP embeddings ----------

    @property
    def embeddings_path(self) -> Path:
        return self.data_root / "embeddings.npy"

    @property
    def embeddings_meta_path(self) -> Path:
        return self.data_root / "embeddings_meta.csv"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "images"

    # ---------- optional pose embeddings ----------

    @property
    def pose_embeddings_path(self) -> Path:
        """
        Optional pose embeddings for artworks.

        Shape must match embeddings.npy (N, D_pose).
        If the file does not exist, PoseMatcher will gracefully fall back
        to CLIP-only matching.
        """
        return self.data_root / "pose_embeddings.npy"
