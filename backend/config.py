# config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Settings:
    """
    Global configuration for the multi-museum backend.
    """

    # === Core paths ===
    data_root: str = "data"                         # Root folder containing museum subfolders
    pose_model_path: str = "models/yolov8n-pose.pt" # Optional; Ultralytics will auto-download if missing

    # === CLIP embedding model ===
    clip_model_name: str = "ViT-B-32"               # Model architecture
    clip_pretrained: str = "openai"                 # Pretrained weights source

    # === Museum selection ===
    allow_museums: Optional[List[str]] = field(default=None)  # Restrict loading to specific subfolders

    # === Matching parameters ===
    topk_default: int = 3                           # Default number of matches returned
    use_dual_view: bool = True                      # Whether to use dual-view (full + crop)
    stillness_threshold_sec: float = 3.5            # Reserved for real-time mode

    # === Logging / Debug ===
    verbose: bool = False                           # Toggle debug output

    # === Device override ===
    device_override: Optional[str] = None           # Manually force device ("cpu", "cuda", "mps")