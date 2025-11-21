# scripts/setup_check.py

from __future__ import annotations
import json, socket
from pathlib import Path

# --------------------------------------------------
# Auto paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_LOCAL = ROOT / "data" / "local"
DATA_MIXED = ROOT / "data" / "mixed"

BACKEND = ROOT / "backend"
MODEL_PT = BACKEND / "model" / "yolov8n-pose.pt"   # <-- your actual path

# --------------------------------------------------
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

# --------------------------------------------------
def check_dataset(folder: Path):
    emb = folder / "embeddings.npy"
    meta = folder / "embeddings_meta.csv"
    pose = folder / "pose_embeddings.npy"
    images = folder / "images"

    return {
        "exists_embeddings.npy": emb.exists(),
        "exists_embeddings_meta.csv": meta.exists(),
        "exists_pose_embeddings.npy": pose.exists(),
        "exists_images_dir": images.is_dir(),
        "images_count": len(list(images.glob("*.*"))) if images.is_dir() else 0
    }

# --------------------------------------------------
def main():
    report = {}

    # Dataset checks
    report["local"] = check_dataset(DATA_LOCAL)
    report["mixed"] = check_dataset(DATA_MIXED)

    # Model weights
    report["model_yolov8n_pose.pt_exists"] = MODEL_PT.exists()

    # Python deps
    check_list = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "open_clip_torch": "open_clip",
        "ultralytics": "ultralytics",
        "streamlit": "streamlit",
        "torch": "torch",
        "pandas": "pandas",
        "numpy": "numpy",
        "requests": "requests",
    }

    deps = {}
    for pkg, import_name in check_list.items():
        try:
            __import__(import_name)
            deps[pkg] = True
        except Exception:
            deps[pkg] = False

    report["deps"] = deps

    # Port check
    report["port_8000_in_use"] = port_in_use(8000)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
