"""
Global configuration for the EAR backend.

This file defines a simple Settings model that knows where your
data lives (embeddings, metadata CSV, images directory, pose
embeddings) and which device to use for inference.
"""

from pathlib import Path
from pydantic import BaseModel


# Project root = repo root (EmbodiedAestheticReconstruction/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    # Root folder that contains embeddings + metadata + images
    # NOTE: we now default to the *global mixed* index (local + met).
    data_root: Path = PROJECT_ROOT / "data" / "mixed"

    # Default museum name used when the frontend does not specify one
    # (you can conceptually still call it "museum", but it's really the mixed index)
    default_museum: str = "mixed"

    # Device: "auto", "cpu", "cuda", "mps"
    device: str = "auto"

    # Weight for pose embeddings when mixing scores:
    # final_score = (1 - pose_weight) * clip_score + pose_weight * pose_score
    pose_weight: float = 0.3

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
