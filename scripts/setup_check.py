# scripts/setup_check.py
from __future__ import annotations
import os, sys, socket, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "local"
MODEL_PT = ROOT / "backend" / "models" / "yolov8n-pose.pt"

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def main():
    report = {}

    # embeddings
    emb = DATA / "embeddings.npy"
    meta = DATA / "embeddings_meta.csv"
    report["embeddings.npy"] = emb.exists()
    report["embeddings_meta.csv"] = meta.exists()
    report["images_dir_exists"] = (DATA / "images").is_dir()
    report["images_count"] = len(list((DATA / "images").glob("*.*"))) if (DATA / "images").is_dir() else 0

    # model weights
    report["yolov8n-pose.pt_exists"] = MODEL_PT.exists()

    # python deps quick check
    deps = {}
    for pkg in ["fastapi", "uvicorn", "open_clip_torch", "ultralytics", "streamlit", "torch", "pandas", "numpy"]:
        try:
            __import__(pkg if pkg != "open_clip_torch" else "open_clip")
            deps[pkg] = True
        except Exception:
            deps[pkg] = False
    report["deps"] = deps

    # ports
    report["port_8000_in_use"] = port_in_use(8000)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
