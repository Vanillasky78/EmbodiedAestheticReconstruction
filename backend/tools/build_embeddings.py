#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_embeddings.py
-----------------------

Build CLIP-style image embeddings for a single museum folder.

Expected layout per museum:

  data/<museum>/
    images/
      *.jpg / *.png ...
    embeddings_meta.csv      # must contain at least a 'filename' column

Output per museum:

  data/<museum>/embeddings.npy     # (N, D) CLIP image embeddings

Typical workflow
----------------

# 1. Prepare per-museum metadata (already done in your project)
#    data/local/embeddings_meta.csv
#    data/met/embeddings_meta.csv
#    data/aic/embeddings_meta.csv
#
# 2. Build embeddings for each museum:
#
#    python backend/tools/build_embeddings.py --museum local --data_root data
#    python backend/tools/build_embeddings.py --museum met   --data_root data
#    python backend/tools/build_embeddings.py --museum aic   --data_root data
#
# 3. Merge them into data/mixed/ with:
#
#    python backend/tools/build_mixed_index.py --museums local met aic --data_root data --out_museum mixed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------


def detect_device(device_override: Optional[str] = None) -> torch.device:
    """
    Auto-detect CUDA / MPS / CPU, with optional manual override.
    """
    if device_override:
        return torch.device(device_override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_clip_model(
    model_name: str,
    pretrained: str,
    device: torch.device,
):
    """
    Load an OpenCLIP image encoder and its preprocessing transform.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a 2D array.
    """
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def resolve_filename(row: pd.Series) -> Optional[str]:
    """
    Try to resolve an image filename from a metadata row.

    We primarily look for a 'filename' column, but fall back to
    common alternatives just in case.
    """
    for col in ["filename", "file", "image_path", "path"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return None


# ----------------------------------------------------------
# Core logic
# ----------------------------------------------------------


def build_for_museum(
    museum: str,
    data_root: str = "data",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
) -> None:
    """
    Build CLIP image embeddings for one museum.

    The function will:

    1. Read data/<museum>/embeddings_meta.csv
    2. For each row, resolve a filename and open the image from data/<museum>/images/
    3. Encode it with CLIP and collect a (N, D) embedding matrix
    4. Save it as data/<museum>/embeddings.npy
    """
    base = Path(data_root) / museum
    img_dir = base / "images"
    meta_path = base / "embeddings_meta.csv"
    out_path = base / "embeddings.npy"

    # --- sanity checks -------------------------------------------------
    if not img_dir.is_dir():
        print(f"[ERROR] images folder not found: {img_dir}")
        return

    if not meta_path.is_file():
        print(f"[ERROR] embeddings_meta.csv not found: {meta_path}")
        return

    df = pd.read_csv(meta_path)
    if df.empty:
        print(f"[WARN] no rows found in meta file: {meta_path}")
        return

    # Try to ensure there is at least some filename-like column
    if not any(col in df.columns for col in ["filename", "file", "image_path", "path"]):
        print(
            f"[ERROR] no filename-like column found in {meta_path}. "
            f"Expected at least one of: filename / file / image_path / path"
        )
        return

    dev = detect_device(device)
    print("-" * 60)
    print(f"[INFO] Building CLIP embeddings for museum='{museum}'")
    print(f"[INFO] data_root   = {data_root}")
    print(f"[INFO] images_dir  = {img_dir}")
    print(f"[INFO] meta_csv    = {meta_path}")
    print(f"[INFO] rows in meta= {len(df)}")
    print(f"[INFO] device      = {dev}")
    print(f"[INFO] model       = {model_name} ({pretrained})")

    model, preprocess = load_clip_model(model_name, pretrained, dev)

    feats: List[np.ndarray] = []
    kept = 0
    skipped_missing = 0
    skipped_error = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=museum):
        fname = resolve_filename(row)
        if not fname:
            skipped_missing += 1
            continue

        img_path = img_dir / fname
        if not img_path.is_file():
            print(f"[WARN] missing image file: {img_path}")
            skipped_missing += 1
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open {img_path}: {e}")
            skipped_error += 1
            continue

        tensor = preprocess(pil).unsqueeze(0).to(dev)

        # autocast is safe for cuda/mps and disabled for cpu
        use_amp = dev.type in ("cuda", "mps")
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=use_amp):
            feat = model.encode_image(tensor)
            feat = feat.float().cpu().numpy()  # (1, D)

        feats.append(feat)
        kept += 1

    if not feats:
        print(f"[WARN] no features produced for museum='{museum}'")
        return

    E = np.concatenate(feats, axis=0)  # (N, D)
    E = normalize_rows(E)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, E)

    print(f"[OK] museum='{museum}': embeddings shape={E.shape} → {out_path}")
    print(
        f"[INFO] stats: kept={kept}, "
        f"missing_files={skipped_missing}, "
        f"open_errors={skipped_error}"
    )
    print("-" * 60)


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP image embeddings for one museum folder."
    )
    parser.add_argument(
        "--museum",
        type=str,
        required=True,
        help="Museum folder name under data/, e.g. local / met / aic / mixed",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory that contains museum folders (default: data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model name (default: ViT-B-32)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained tag for OpenCLIP (default: openai)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, e.g. 'cuda', 'cpu', 'mps'",
    )
    args = parser.parse_args()

    build_for_museum(
        museum=args.museum,
        data_root=args.data_root,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )


if __name__ == "__main__":
    main()
