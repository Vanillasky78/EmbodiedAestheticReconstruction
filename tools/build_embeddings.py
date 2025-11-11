# tools/build_embeddings.py
# -----------------------------------------------------------------------------
# Build CLIP embeddings for images under data/<museum>/images/
# Outputs:
#   - data/<museum>/embeddings.npy
#   - data/<museum>/embeddings_meta.csv  (at least a `filename` column)
# Optionally merges with an external metadata CSV by `filename`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import csv
import glob
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
import open_clip

# If you placed this script in tools/ and run from project root, the imports are fine.
# If you run it from tools/, it still works because we don't import project modules here.


# ----------------------------- Utilities -------------------------------------


def discover_images(img_dir: str, exts: List[str]) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    files = sorted(set(files))
    return files


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smart_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (norm + eps)


def load_external_meta(meta_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load an external CSV (UTF-8) and map by filename (case-sensitive).
    Assumes there is a column named `filename`.
    """
    table: Dict[str, Dict[str, str]] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames:
            raise ValueError("External metadata CSV must contain a 'filename' column.")
        for row in reader:
            fn = row["filename"]
            table[fn] = row
    return table


# ----------------------------- Core Builder ----------------------------------


def build_one_museum(
    museum_folder: str,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device_str: str | None = None,
    batch_size: int = 32,
    exts: List[str] | None = None,
    limit: int | None = None,
    external_meta_csv: str | None = None,
    overwrite: bool = True,
) -> None:
    """
    Build embeddings for a single museum folder.

    Folder structure:
        museum_folder/
            images/
            embeddings.npy              (output)
            embeddings_meta.csv         (output)

    Args:
        museum_folder: path like 'data/met'
        external_meta_csv: optional CSV to merge by `filename`
    """
    exts = exts or [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    museum_name = os.path.basename(museum_folder.rstrip("/"))

    img_dir = os.path.join(museum_folder, "images")
    if not os.path.isdir(img_dir):
        print(f"[WARN] skip {museum_name}: no images/ found at {img_dir}")
        return

    out_npy = os.path.join(museum_folder, "embeddings.npy")
    out_csv = os.path.join(museum_folder, "embeddings_meta.csv")
    if (os.path.exists(out_npy) or os.path.exists(out_csv)) and not overwrite:
        print(f"[SKIP] {museum_name}: outputs already exist and overwrite=False")
        return

    files = discover_images(img_dir, exts)
    if not files:
        print(f"[WARN] {museum_name}: no images with {exts} under {img_dir}")
        return
    if limit and limit > 0:
        files = files[:limit]

    device = smart_device(device_str)
    print(f"[INFO] museum={museum_name} | images={len(files)} | device={device} | model={clip_model}/{clip_pretrained}")

    # Create model
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    model.eval()

    feats = []
    meta_rows = []

    # Batch inference
    n = len(files)
    steps = (n + batch_size - 1) // batch_size
    with torch.no_grad():
        for step in range(steps):
            chunk = files[step * batch_size : (step + 1) * batch_size]
            images = []
            good_paths = []
            for path in chunk:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(preprocess(img))
                    good_paths.append(path)
                except Exception as e:
                    print(f"[SKIP] {path}: {e}")

            if not images:
                continue

            batch = torch.stack(images, dim=0).to(device)
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.detach().cpu().numpy().astype(np.float32)

            feats.append(emb)
            meta_rows.extend([{"filename": os.path.basename(p)} for p in good_paths])

            print(f"[{museum_name}] progress: {min((step + 1) * batch_size, n)}/{n}")

    if not feats:
        print(f"[WARN] {museum_name}: no valid images processed.")
        return

    embs = np.concatenate(feats, axis=0)
    embs = l2_normalize(embs, axis=1)

    # Save embeddings
    np.save(out_npy, embs)

    # Merge external metadata if provided
    if external_meta_csv and os.path.exists(external_meta_csv):
        try:
            ext_map = load_external_meta(external_meta_csv)
            # enrich rows
            enriched = []
            # collect all headers
            all_keys = set(["filename"])
            for r in meta_rows:
                fn = r["filename"]
                extra = ext_map.get(fn, {})
                merged = dict(extra)
                merged["filename"] = fn  # ensure filename is present and last write
                enriched.append(merged)
                all_keys.update(merged.keys())
            fieldnames = list(all_keys)
            # write CSV
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in enriched:
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARN] failed to merge external metadata: {e}")
            # fallback to minimal csv
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename"])
                writer.writeheader()
                for row in meta_rows:
                    writer.writerow(row)
    else:
        # minimal CSV (filename only)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"])
            writer.writeheader()
            for row in meta_rows:
                writer.writerow(row)

    print(f"[OK] {museum_name}: embeddings={embs.shape} → {out_npy}")
    print(f"[OK] {museum_name}: meta rows={len(meta_rows)} → {out_csv}")


def build_all(
    root: str,
    **kwargs,
) -> None:
    """Iterate all subfolders under root and build embeddings for each."""
    subs = sorted([p for p in glob.glob(os.path.join(root, "*/"))])
    if not subs:
        print(f"[WARN] no subfolders under {root}")
        return
    for folder in subs:
        build_one_museum(folder, **kwargs)


# ----------------------------- Entry Point -----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build CLIP embeddings for data/<museum>/images/"
    )
    p.add_argument("--root", default="data", help="Root folder containing museum subfolders")
    p.add_argument("--museum", default=None, help="Build only for a specific subfolder name (e.g., 'met')")
    p.add_argument("--all", action="store_true", help="Process all subfolders under --root")
    p.add_argument("--clip-model", default="ViT-B-32", help="CLIP model name (e.g., ViT-B-32, ViT-L-14)")
    p.add_argument("--clip-pretrained", default="openai", help="Pretrained weights (e.g., openai, laion2b_s34b_b79k)")
    p.add_argument("--device", default=None, help="Force device: cuda | mps | cpu (auto if omitted)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding inference")
    p.add_argument("--exts", default=".jpg,.jpeg,.png,.webp,.bmp", help="Comma-separated image extensions")
    p.add_argument("--limit", type=int, default=None, help="Only process first N images (debug)")
    p.add_argument("--meta", default=None, help="External metadata CSV to merge by `filename`")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main():
    args = parse_args()
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    kwargs = dict(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device_str=args.device,
        batch_size=args.batch_size,
        exts=exts,
        limit=args.limit,
        external_meta_csv=args.meta,
        overwrite=args.overwrite,
    )

    if args.all:
        build_all(args.root, **kwargs)
    else:
        if not args.museum:
            raise SystemExit("Please set --museum=NAME or use --all.")
        folder = os.path.join(args.root, args.museum)
        if not os.path.isdir(folder):
            raise SystemExit(f"Folder not found: {folder}")
        build_one_museum(folder, **kwargs)


if __name__ == "__main__":
    main()
