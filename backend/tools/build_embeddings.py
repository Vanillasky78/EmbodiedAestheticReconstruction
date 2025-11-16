#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build pose embeddings for a museum folder.

For now this uses the same CLIP encoder as build_embeddings.py,
so everything works out-of-the-box. Later you can replace it with a
true pose encoder (e.g. from YOLO keypoints).

Layout (per museum):
  data/<museum>/
    images/
    pose_embeddings.npy      # (N, D_pose)  (same N as embeddings.npy)

Usage example:
  python backend/tools/build_pose_embeddings.py --museum mixed --data_root data

Deps:
  pip install open_clip_torch torch pillow numpy tqdm
"""

import argparse
import os
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


def detect_device(device_override: Optional[str] = None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
            files.append(p)
    return files


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def build_pose_for_museum(
    museum: str,
    data_root: str = "data",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    overwrite: bool = False,
):
    base = os.path.join(data_root, museum)
    img_dir = os.path.join(base, "images")
    pose_path = os.path.join(base, "pose_embeddings.npy")

    if not os.path.isdir(img_dir):
        print(f"[WARN] images folder not found: {img_dir}")
        return

    if os.path.exists(pose_path) and not overwrite:
        print(f"[SKIP] {museum}: pose_embeddings already exists ({pose_path})")
        return

    files = list_images(img_dir)
    if not files:
        print(f"[WARN] no images in {img_dir}")
        return

    dev = detect_device(device)
    print(f"[INFO] building pose embeddings for {museum} on {dev}…")

    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=dev,
    )
    model.eval()

    feats = []
    for path in tqdm(files, desc=f"{museum}"):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open {path}: {e}")
            continue

        im = preprocess(pil).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(device_type=str(dev), enabled=(dev.type in ["cuda", "mps"])):
            feat = model.encode_image(im)
            feat = feat.float().cpu().numpy()
        feats.append(feat)

    if not feats:
        print(f"[WARN] no features produced for: {museum}")
        return

    E = np.concatenate(feats, axis=0)  # (N, D)
    E = normalize_rows(E)
    np.save(pose_path, E)
    print(f"[OK] {museum}: pose_embeddings={E.shape} → {pose_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--museum", type=str, help="Museum folder name under data/, e.g. local / met / mixed")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.museum:
        print("\nPlease set --museum=NAME (e.g. --museum=mixed).\n")
        raise SystemExit(1)

    build_pose_for_museum(
        museum=args.museum,
        data_root=args.data_root,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
