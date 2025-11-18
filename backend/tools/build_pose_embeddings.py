#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_pose_embeddings.py
------------------------

Offline script to build pose_embeddings.npy for a given museum folder.

Expected layout:

    data/<museum>/
        images/                 # artwork images
        embeddings_meta.csv     # at least has a "filename" column

This script will:

  1. Load YOLOv8-Pose.
  2. For each row in embeddings_meta.csv:
       - open the corresponding image
       - run YOLOv8-Pose to get keypoints
       - convert keypoints -> 1D pose vector (encode_keypoints_to_pose_vector)
  3. Save:
       data/<museum>/pose_embeddings.npy   (N, D_pose)
       data/<museum>/pose_meta.csv         (rows that had valid pose vectors)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

# Import our pose encoder helper
from backend.utils_pose import encode_keypoints_to_pose_vector


# ------------------------------------------------------------
# Helper: run YOLOv8-Pose and encode to 1D pose vector
# ------------------------------------------------------------
def extract_pose_vector(
    pil_img: Image.Image,
    model: YOLO,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Run YOLOv8-Pose on a PIL image and convert keypoints to a 1D pose vector.

    Returns
    -------
    pose_vec : np.ndarray or None
        1D numpy array (D_pose,) if detection succeeded, else None.
    """
    rgb = np.array(pil_img.convert("RGB"))

    # Always run YOLO on the given device (we typically choose CPU for MPS safety)
    results = model.predict(rgb, imgsz=640, device=str(device), verbose=False)

    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None

    kpts = results[0].keypoints

    # (17, 2) keypoint coordinates
    xy = kpts.xy[0].cpu().numpy().astype("float32")

    # Confidence can be None on some Ultralytics versions → fall back to all-ones
    if getattr(kpts, "conf", None) is not None:
        conf = kpts.conf[0].cpu().numpy().astype("float32")  # (17,)
    else:
        conf = np.ones(xy.shape[0], dtype="float32")

    h, w = rgb.shape[:2]
    xy_norm = xy.copy()
    xy_norm[:, 0] /= float(w)
    xy_norm[:, 1] /= float(h)

    # Our project’s helper: (xy_norm, conf) -> 1D vector
    pose_vec_1d = encode_keypoints_to_pose_vector(xy_norm, conf)
    if pose_vec_1d is None:
        return None

    return pose_vec_1d.astype("float32")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build pose_embeddings.npy for a museum folder.")
    parser.add_argument(
        "--museum_dir",
        type=str,
        required=True,
        help="Path to museum directory, e.g. data/aic or data/local",
    )
    parser.add_argument(
        "--yolo",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLOv8-Pose weights file (relative or absolute path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",  # safer than MPS for pose on Apple
        help='Device to run YOLO on: "cpu", "cuda", or "mps". Default = "cpu".',
    )

    args = parser.parse_args()

    museum_dir = Path(args.museum_dir).resolve()
    if not museum_dir.exists():
        raise FileNotFoundError(f"Museum dir not found: {museum_dir}")

    meta_csv = museum_dir / "embeddings_meta.csv"
    images_dir = museum_dir / "images"

    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    print(f"[INFO] Loaded metadata from: {meta_csv}")
    print(f"[INFO] Images dir: {images_dir}")

    # -------- device --------
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        print("[WARN] MPS requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # -------- YOLO model --------
    yolo_path = Path(args.yolo)
    if not yolo_path.exists():
        # also try museum_dir parent / repo-root
        repo_root = museum_dir.parents[2]  # EmbodiedAestheticReconstruction/
        alt = repo_root / "yolov8n-pose.pt"
        if alt.exists():
            yolo_path = alt
        else:
            raise FileNotFoundError(f"YOLO weights not found: {args.yolo}")

    print(f"[INFO] Loading YOLO model: {yolo_path}")
    model = YOLO(str(yolo_path))

    # -------- read metadata rows --------
    rows: List[Dict] = []
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"[INFO] Total rows in metadata: {len(rows)}")

    pose_vectors: List[np.ndarray] = []
    kept_rows: List[Dict] = []

    for i, r in enumerate(rows):
        fname = (
            r.get("filename")
            or r.get("file")
            or r.get("image_path")
            or r.get("path")
        )
        if not fname:
            continue

        img_path = Path(fname)
        if not img_path.is_absolute():
            img_path = images_dir / img_path

        if not img_path.exists():
            print(f"[WARN] Missing image file: {img_path}")
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Failed to open {img_path}: {exc}")
            continue

        pose_vec = extract_pose_vector(pil, model, device)
        if pose_vec is None:
            # no reliable person / keypoints in this image
            continue

        pose_vectors.append(pose_vec)
        kept_rows.append(r)

        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i+1}/{len(rows)} rows… kept {len(kept_rows)}")

    if len(pose_vectors) == 0:
        raise RuntimeError("No pose vectors generated. Check YOLO model / images.")

    pose_arr = np.stack(pose_vectors, axis=0)
    out_pose = museum_dir / "pose_embeddings.npy"
    np.save(out_pose, pose_arr)
    print(f"[OK] Saved pose embeddings: {out_pose} (shape={pose_arr.shape})")

    out_csv = museum_dir / "pose_meta.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = None
        for r in kept_rows:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(r.keys()))
                writer.writeheader()
            writer.writerow(r)

    print(f"[OK] Saved pose metadata: {out_csv} (rows={len(kept_rows)})")


if __name__ == "__main__":
    main()
