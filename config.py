# config.py
# -----------------------------------------------------------------------------
# Embodied Aesthetic Reconstruction — Configuration File
# -----------------------------------------------------------------------------
# This module defines global paths and runtime settings
# for the FastAPI multi-museum backend.
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    """
    Global configuration for EAR-API.
    Modify these paths or add more attributes if needed.
    """

    # === Core paths ===
    data_root: str = "data"                         # Root folder containing museum subfolders
    pose_model_path: str = "models/yolov8n-pose.pt" # Optional; Ultralytics will auto-download if missing

    # === CLIP embedding model ===
    clip_model_name: str = "ViT-B-32"               # Model architecture (ViT-B/32 is lightweight and good)
    clip_pretrained: str = "openai"                 # Pretrained weights source (openai, laion2b, etc.)

    # === Museum selection ===
    # Set to a list of folder names under data/ to restrict loading;
    # Leave None to automatically load all available subfolders.
    allow_museums: Optional[List[str]] = field(default=None)

    # === Matching parameters ===
    topk_default: int = 3                           # Default number of matches returned
    use_dual_view: bool = True                      # Use full-image + center-crop dual-view embedding
    stillness_threshold_sec: float = 3.5            # (Reserved) threshold for real-time pose capture stability

    # === Logging / Debug ===
    verbose: bool = False                           # Toggle debug logging for model loading, etc.

    # === Device control (optional override) ===
    # Leave empty for automatic detection (CUDA / MPS / CPU)
    device_override: Optional[str] = None

    # === Future expansion ===
    # Example: add API keys or URLs if you fetch museum data from open APIs
    # met_api_key: str = ""
    # rijks_api_key: str = ""
