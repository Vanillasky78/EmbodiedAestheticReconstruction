"""
Global configuration for the EAR backend.

This file defines a simple Settings model that knows where your
data lives (embeddings, metadata CSV, images directory) and which
device to use for inference.
"""

from pathlib import Path
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    # 使用混合索引：local + met
    data_root: Path = PROJECT_ROOT / "data" / "mixed"

    # 默认“馆”名随便写，但统一一下
    default_museum: str = "mixed"

    # Device: "auto", "cpu", "cuda", "mps"
    device: str = "auto"


    # Device: "auto", "cpu", "cuda", "mps"
    device: str = "auto"

    class Config:
        arbitrary_types_allowed = True

    @property
    def embeddings_path(self) -> Path:
        return self.data_root / "embeddings.npy"

    @property
    def embeddings_meta_path(self) -> Path:
        return self.data_root / "embeddings_meta.csv"

    @property
    def images_dir(self) -> Path:
        return self.data_root / "images"