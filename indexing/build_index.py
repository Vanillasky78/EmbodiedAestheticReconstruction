"""
Builds an image–text embedding index (CLIP + FAISS)
for the Embodied Aesthetic Reconstruction project.

This script reads a metadata JSONL file (e.g. data/images/metadata.jsonl),
locates local images, encodes them with OpenAI CLIP (via open_clip),
and saves a FAISS index for fast retrieval.

Usage:
    python -m indexing.build_index \
        --dataset_jsonl data/images/metadata.jsonl \
        --images_dir data/images \
        --out_dir indexing
"""

import os
import sys
import json
import argparse
from pathlib import Path
from glob import escape as glob_escape
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import open_clip
import faiss


# ---------- constants ----------
EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# ---------- helper functions ----------

def find_local_image(record: dict, images_dir: Path) -> str | None:
    """
    Find the corresponding image file for a metadata record.
    Tries multiple matching methods in order of reliability:
        1. Direct path if exists
        2. objectID.jpg or artwork_id.jpg
        3. Escaped fuzzy match (artist + title)
        4. Fallback: by source prefix
    All glob patterns are escaped to avoid Windows errors.
    """
    # Direct path
    p_given = record.get("image_path")
    if p_given and Path(p_given).exists():
        return str(p_given)

    # Try objectID.jpg
    oid = record.get("objectID") or record.get("artwork_id")
    if oid is not None:
        p_oid = images_dir / f"{oid}.jpg"
        if p_oid.exists():
            return str(p_oid)

    # Safe fuzzy match
    artist = str(record.get("artistDisplayName") or record.get("artist_name_en") or "").strip()
    title = str(record.get("title") or record.get("artwork_title_en") or "").strip()

    safe_artist = glob_escape(artist[:15].replace(" ", "_"))
    safe_title = glob_escape(title[:15].replace(" ", "_"))

    try:
        for p in images_dir.glob(f"*{safe_artist}*{safe_title}*"):
            if p.suffix.lower() in EXTS:
                return str(p)
    except ValueError:
        pass

    # Fallback: source prefix
    src = str(record.get("source_api") or "").strip()
    if src:
        try:
            for p in images_dir.glob(f"{glob_escape(src)}_*"):
                if p.suffix.lower() in EXTS:
                    return str(p)
        except ValueError:
            pass

    return None


def load_records(jsonl_path: Path):
    """Load JSONL metadata and attach local image paths."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            records.append(rec)
    print(f"[meta] loaded {len(records)} records from {jsonl_path}")
    return records


def encode_images(records, images_dir: Path, model, preprocess, device):
    """Compute CLIP embeddings for local images."""
    embeddings = []
    valid_records = []

    for rec in tqdm(records, desc="Encoding images"):
        img_path = find_local_image(rec, images_dir)
        if not img_path:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                tensor = preprocess(img).unsqueeze(0).to(device)
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy())
                rec["image_path"] = img_path
                valid_records.append(rec)
        except Exception:
            continue

    if not embeddings:
        raise RuntimeError("No valid images found for embedding!")

    return np.vstack(embeddings), valid_records


def save_index(embeddings: np.ndarray, out_dir: Path, meta_records: list):
    """Save FAISS index + ids + meta info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "ids.npy", np.arange(len(meta_records)))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)

    print(f"[index] saved {len(meta_records)} embeddings → {out_dir}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", required=True, type=str)
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    jsonl_path = Path(args.dataset_jsonl)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    # Load records
    records = load_records(jsonl_path)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    print(f"[clip] using device={device}")

    # Encode
    embeddings, valid_records = encode_images(records, images_dir, model, preprocess, device)

    # Save index
    save_index(embeddings, out_dir, valid_records)


if __name__ == "__main__":
    main()
