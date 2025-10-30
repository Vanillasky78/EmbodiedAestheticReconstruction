#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
indexing/build_index.py
Usage:
  python indexing/build_index.py \
    --dataset_jsonl data/interim/portrait_art_dataset.jsonl \
    --images_dir data/images \
    --out_dir indexing
"""

import argparse, json, os
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from PIL import Image

import torch
import open_clip


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_model(device: str = "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def encode_images(rows: List[Dict], images_dir: Path, model, preprocess, device: str):
    """返回 (embeddings[N, D], ids[N], meta[N])"""
    embs = []
    ids = []
    meta = []

    for r in rows:
        # 兼容不同字段
        rid = r.get("artwork_id") or r.get("objectID") or r.get("id")
        img_rel = r.get("image_path")  # 如果 JSONL 已经写了本地路径
        if not img_rel:
            # 尝试用文件名或 id 猜测图片路径（可按你的命名规则再加几种）
            # 例如：images/{artwork_id}.jpg
            img_rel = f"{rid}.jpg"

        img_path = images_dir / img_rel
        if not img_path.exists():
            # 没图就跳过（也可以只做文字/其他来源嵌入，这里从简）
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy())
            ids.append(str(rid))
            # 只保留渲染需要的关键字段
            meta.append({
                "artwork_id": str(rid),
                "title": r.get("artwork_title_en") or r.get("title"),
                "artist": r.get("artist_name_en") or r.get("artistDisplayName"),
                "year": r.get("year") or r.get("objectDate"),
                "museum": r.get("museum") or r.get("repository") or r.get("department"),
                "license": r.get("license") or ("Public Domain" if r.get("isPublicDomain") else None),
                "image_path": str(img_path),
            })
        except Exception as e:
            print(f"[skip] {img_path}: {e}")

    if not embs:
        raise RuntimeError("No embeddings created. Make sure images exist under --images_dir.")

    embs = np.vstack(embs).astype("float32")  # (N, D)
    return embs, np.array(ids), meta


def build_faiss(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # 余弦等价于内积(已单位化)
    index.add(embs)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", type=str, required=True)
    ap.add_argument("--images_dir", type=str, default="data/images")
    ap.add_argument("--out_dir", type=str, default="indexing")
    args = ap.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"{dataset_jsonl} not found")
    if not images_dir.exists():
        print(f"[warn] images_dir {images_dir} not found. You may get 0 embeddings.")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

    rows = load_jsonl(dataset_jsonl)
    print(f"[data] rows: {len(rows)}")

    model, preprocess = build_model(device=device)
    embs, ids, meta = encode_images(rows, images_dir, model, preprocess, device)

    index = build_faiss(embs)

    faiss_path = out_dir / "faiss.index"
    ids_path   = out_dir / "ids.npy"
    meta_path  = out_dir / "meta.json"

    faiss.write_index(index, str(faiss_path))
    np.save(str(ids_path), ids, allow_pickle=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] saved: {faiss_path}\n[ok] saved: {ids_path}\n[ok] saved: {meta_path}")


if __name__ == "__main__":
    main()
