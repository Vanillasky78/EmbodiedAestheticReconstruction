# scripts/setup_check.py

from __future__ import annotations
import os, sys, socket, json
from pathlib import Path

# --------------------------------------------------
# Paths (AUTO-DETECTED based on your project layout)
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_LOCAL = ROOT / "data" / "local"
DATA_MIXED = ROOT / "data" / "mixed"

BACKEND = ROOT / "backend"
MODEL_PT = BACKEND / "model" / "yolov8n-pose.pt"

# --------------------------------------------------
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    report = {}

    # --------------------------------------------------
    # Dataset checks: LOCAL + MIXED
    # --------------------------------------------------
    for name, folder in {
        "local": DATA_LOCAL,
        "mixed": DATA_MIXED,
    }.items():

        emb = folder / "embeddings.npy"
        meta = folder / "embeddings_meta.csv"
        pose_emb = folder / "pose_embeddings.npy"
        images = folder / "images"

        report[f"{name}_embeddings.npy_exists"] = emb.exists()
        report[f"{name}_embeddings_meta.csv_exists"] = meta.exists()
        report[f"{name}_pose_embeddings.npy_exists"] = pose_emb.exists()
        report[f"{name}_images_dir_exists"] = images.is_dir()
        report[f"{name}_images_count"] = len(list(images.glob("*.*"))) if images.is_dir() else 0

    # --------------------------------------------------
    # Model weight
    # --------------------------------------------------
    report["yolov8n-pose.pt_exists"] = MODEL_PT.exists()

    # --------------------------------------------------
    # Python dependencies
    # --------------------------------------------------
    deps = {}
    for pkg, import_name in {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "open_clip_torch": "open_clip",
        "ultralytics": "ultralytics",
        "streamlit": "streamlit",
        "torch": "torch",
        "pandas": "pandas",
        "numpy": "numpy",
    }.items():

        try:
            __import__(import_name)
            deps[pkg] = True
        except Exception:
            deps[pkg] = False

    report["deps"] = deps

    # --------------------------------------------------
    # Ports
    # --------------------------------------------------
    report["port_8000_in_use"] = port_in_use(8000)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
