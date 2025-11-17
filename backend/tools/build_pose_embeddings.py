"""
build_pose_embeddings.py
-------------------------
Generate pose-based embeddings for all artworks using YOLOv8-Pose.

Output:
    data/mixed/pose_embeddings.npy
    (same row ordering as embeddings_meta.csv)

Usage:
    conda activate ear-mvp
    python backend/tools/build_pose_embeddings.py --data-root data/mixed
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# YOLOv8-Pose
try:
    from ultralytics import YOLO
except Exception as exc:
    raise RuntimeError("Ultralytics YOLO is required. `pip install ultralytics`.") from exc


# -------------------------
# Pose vector utilities
# -------------------------

def encode_pose_keypoints(kps: np.ndarray | None, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert raw keypoints (N x 3: x,y,confidence) into a fixed pose vector.

    We encode:
      - normalized coordinates (x/img_w, y/img_h)
      - per-joint existence mask
      - joint angle features
    """

    num_joints = 17  # YOLOv8 Pose has 17 keypoints (0..16)
    if kps is None:
        # No detection → zero vector
        return np.zeros(num_joints * 4, dtype=np.float32)

    # If YOLO returns more than one person, we take the one with largest area
    # but build_embeddings will only run on artwork images (static), so 1 person is expected.
    if len(kps.shape) == 3:  # shape [num_people, 17, 3]
        kps = kps[0]

    # Prepare vector components
    coord_vec = np.zeros((num_joints, 2), dtype=np.float32)
    mask_vec  = np.zeros((num_joints, 1), dtype=np.float32)
    angle_vec = np.zeros((num_joints, 1), dtype=np.float32)

    # Normalize
    for i in range(num_joints):
        x, y, conf = kps[i]
        if conf > 0.1:
            coord_vec[i] = np.array([x / img_w, y / img_h])
            mask_vec[i] = 1.0
        else:
            coord_vec[i] = 0.0
            mask_vec[i] = 0.0

    # Compute simple limb angles (example: shoulders/elbows/wrists)
    # We encode angle at elbow and knee joints for stability
    def safe_angle(a, b, c):
        """Return angle ABC in radians."""
        if (a is None) or (b is None) or (c is None):
            return 0.0
        ax, ay = a
        bx, by = b
        cx, cy = c
        v1 = np.array([ax - bx, ay - by])
        v2 = np.array([cx - bx, cy - by])
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cosv))  # radians

    # Joint indices follow COCO:
    # 5: left shoulder, 6: right shoulder
    # 7: left elbow,    8: right elbow
    # 9: left wrist,   10: right wrist
    # 11: left hip,    12: right hip
    # 13: left knee,   14: right knee
    # 15: left ankle,  16: right ankle

    joints = kps[:, :2]
    def jp(idx):
        x, y, conf = kps[idx]
        return (x, y) if conf > 0.1 else None

    # elbow angles
    angle_vec[7]  = safe_angle(jp(5), jp(7), jp(9))
    angle_vec[8]  = safe_angle(jp(6), jp(8), jp(10))

    # knee angles
    angle_vec[13] = safe_angle(jp(11), jp(13), jp(15))
    angle_vec[14] = safe_angle(jp(12), jp(14), jp(16))

    # Flatten
    return np.concatenate([
        coord_vec.reshape(-1),
        mask_vec.reshape(-1),
        angle_vec.reshape(-1),
    ]).astype(np.float32)


# -------------------------
# Main build function
# -------------------------

def build_pose_embeddings(data_root: Path, model_path: Path):
    meta_path = data_root / "embeddings_meta.csv"
    images_dir = data_root / "images"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    # Load metadata rows in order
    import csv
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Loaded {len(rows)} metadata rows.")

    # Init YOLO pose model
    print(f"[INFO] Loading YOLOv8-Pose: {model_path}")
    model = YOLO(str(model_path))

    pose_vectors = []

    for row in tqdm(rows, desc="Building pose vectors"):
        fname = (
            row.get("filename")
            or row.get("image_path")
            or row.get("path")
            or row.get("file")
        )
        if not fname:
            pose_vectors.append(np.zeros(17 * 4, dtype=np.float32))
            continue

        img_path = images_dir / fname
        if not img_path.exists():
            pose_vectors.append(np.zeros(17 * 4, dtype=np.float32))
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            pose_vectors.append(np.zeros(17 * 4, dtype=np.float32))
            continue

        w, h = img.size
        img_np = np.array(img)

        # YOLO pose inference
        res = model.predict(img_np, imgsz=640, device="cpu", verbose=False)
        if len(res) == 0 or res[0].keypoints is None:
            pose_vectors.append(np.zeros(17 * 4, dtype=np.float32))
            continue

        kps = res[0].keypoints.cpu().numpy()  # shape: [1, 17, 3]
        vec = encode_pose_keypoints(kps, w, h)
        pose_vectors.append(vec)

    pose_vectors = np.stack(pose_vectors, axis=0)
    out_path = data_root / "pose_embeddings.npy"

    np.save(out_path, pose_vectors)
    print(f"[OK] Saved pose embeddings → {out_path}")
    print(f"[INFO] shape = {pose_vectors.shape} (rows match embeddings_meta.csv)")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/mixed",
                        help="path to dataset folder that contains images/ and embeddings_meta.csv")
    parser.add_argument("--yolo-model", type=str, default="frontend/yolov8n-pose.pt",
                        help="path to YOLOv8-Pose .pt file")

    args = parser.parse_args()

    build_pose_embeddings(
        Path(args.data_root),
        Path(args.yolo_model),
    )
