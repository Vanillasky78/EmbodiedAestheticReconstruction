#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CLIP embeddings for a museum folder.

Layout (per museum):
  data/<museum>/
    images/                  # *.jpg / *.png ...
    embeddings.npy           # (N, D)
    embeddings_meta.csv      # at least: filename + optional columns from --meta

Usage:
  python backend/tools/build_embeddings.py --museum local --meta data/local/portrait_works.csv --overwrite
  python backend/tools/build_embeddings.py --all --overwrite

Deps:
  pip install open_clip_torch torch pillow numpy pandas tqdm
"""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


# ------------------------- utils -------------------------

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


def load_external_meta(meta_csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Robust CSV reader (BOM-safe, header-trim, filename normalization)."""
    if not meta_csv_path:
        return None
    if not os.path.exists(meta_csv_path):
        print(f"[WARN] metadata CSV not found: {meta_csv_path}")
        return None

    df = pd.read_csv(meta_csv_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # accept 'filename' or common variants
    if "filename" not in df.columns:
        alt = None
        for c in ("image", "file"):
            if c in df.columns:
                alt = c
                break
        if alt:
            df = df.rename(columns={alt: "filename"})

    if "filename" not in df.columns:
        raise ValueError("External metadata CSV must contain a 'filename' (or 'image'/'file') column.")

    df["filename"] = df["filename"].map(lambda x: os.path.basename(str(x)).strip())
    return df


def dual_views(preprocess, pil: Image.Image):
    """full image + center crop (square) as a tiny augmentation ensemble."""
    # full
    im_full = preprocess(pil.convert("RGB"))
    # center crop to min side (square), then preprocess again
    w, h = pil.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    pil_square = pil.crop((left, top, left + m, top + m))
    im_square = preprocess(pil_square.convert("RGB"))
    return im_full, im_square


# ------------------------- core -------------------------

def build_for_museum(
    museum: str,
    data_root: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[str] = None,
    meta_csv: Optional[str] = None,
    overwrite: bool = False,
    use_dual_view: bool = True,
):
    out_dir = os.path.join(data_root, museum)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"[WARN] images folder not found: {img_dir}")
        return

    files = list_images(img_dir)
    if not files:
        print(f"[WARN] no images found in: {img_dir}")
        return

    dev = detect_device(device)
    print(f"[INFO] museum={museum} | images={len(files)} | device={dev} | model={model_name}/{pretrained}")

    embed_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "embeddings_meta.csv")

    if (os.path.exists(embed_path) or os.path.exists(meta_path)) and not overwrite:
        print(f"[SKIP] embeddings/meta already exist (use --overwrite to rebuild): {out_dir}")
        return

    # model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=dev)
    model.eval()

    # encode
    feats = []
    for p in tqdm(files, desc=f"[{museum}] progress"):
        try:
            pil = Image.open(p)
        except Exception:
            print(f"[WARN] failed to open image: {p}")
            continue

        if use_dual_view:
            im1, im2 = dual_views(preprocess, pil)
            ims = torch.stack([im1, im2], dim=0).to(dev)
        else:
            ims = torch.stack([preprocess(pil.convert("RGB"))], dim=0).to(dev)

        with torch.no_grad(), torch.autocast(device_type=str(dev), enabled=(dev.type in ["cuda", "mps"])):
            feat = model.encode_image(ims)  # (k, D)
            feat = feat.float().cpu().numpy()
            feat = feat.mean(axis=0, keepdims=True)  # simple average for dual view
        feats.append(feat)

    if not feats:
        print(f"[WARN] no features produced for: {museum}")
        return

    E = np.concatenate(feats, axis=0)  # (N, D)
    E = normalize_rows(E)
    np.save(embed_path, E)
    print(f"[OK] {museum}: embeddings={E.shape} → {embed_path}")

    # base meta (filename only)
    meta_rows = [{"filename": os.path.basename(p)} for p in files]
    base_df = pd.DataFrame(meta_rows)

    # external meta
    ext_df = None
    try:
        ext_df = load_external_meta(meta_csv)
    except Exception as e:
        print(f"[WARN] failed to read external metadata: {e}")

    merged = base_df.merge(ext_df, on="filename", how="left") if ext_df is not None else base_df

    # filename first
    cols = ["filename"] + [c for c in merged.columns if c != "filename"]
    merged = merged[cols]
    merged.to_csv(meta_path, index=False, encoding="utf-8")
    print(f"[OK] {museum}: meta rows={len(merged)} → {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--museum", type=str, help="Museum folder name under data/, e.g., local / met")
    parser.add_argument("--all", action="store_true", help="Build for all subfolders under data/")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default=None, help="cuda / mps / cpu (auto if omitted)")
    parser.add_argument("--meta", type=str, default=None, help="Optional external CSV to merge (must have filename)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_dual", action="store_true", help="Disable dual-view averaging")
    args = parser.parse_args()

    if not args.museum and not args.all:
        print("\nPlease set --museum=NAME or use --all.\n")
        sys.exit(1)

    if args.all:
        for m in sorted(d for d in os.listdir(args.data_root)
                        if os.path.isdir(os.path.join(args.data_root, d))):
            build_for_museum(
                museum=m,
                data_root=args.data_root,
                model_name=args.model,
                pretrained=args.pretrained,
                device=args.device,
                meta_csv=args.meta,
                overwrite=args.overwrite,
                use_dual_view=not args.no_dual,
            )
    else:
        build_for_museum(
            museum=args.museum,
            data_root=args.data_root,
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            meta_csv=args.meta,
            overwrite=args.overwrite,
            use_dual_view=not args.no_dual,
        )


if __name__ == "__main__":
    main()
