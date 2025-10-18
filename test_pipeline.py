"""
Quick pipeline test: load CLIP + FAISS, run retrieval on one image.

Usage:
    python test_pipeline.py --img path/to/input.jpg
Options:
    --index_dir   indexing directory containing faiss.index / ids.npy / meta.json
    --topk        number of results to print (default: 5)
    --k           FAISS shortlist size before re-ranking (default: 200)
    --w_clip      weight for CLIP similarity (default: 0.6)
    --w_pose      weight for pose similarity  (default: 0.3)
    --w_color     weight for color similarity (default: 0.1)
    --require_pd  require Public Domain/CC0 (default: False)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image

# ANN / model
import faiss
import torch
import open_clip

# our modules
from processors.retrieval import Matcher


def load_clip(device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    return model, preprocess, device


def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    ids = np.load(str(index_dir / "ids.npy"), allow_pickle=True)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    if len(ids) != len(meta):
        print(f"[warn] ids ({len(ids)}) and meta ({len(meta)}) length mismatch")
    return index, ids, meta


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Input image for testing")
    ap.add_argument("--index_dir", default="indexing", help="Directory with FAISS index and metadata")
    ap.add_argument("--topk", type=int, default=5, help="Top-N results to print")
    ap.add_argument("--k", type=int, default=200, help="FAISS shortlist size before re-ranking")
    ap.add_argument("--w_clip", type=float, default=0.6, help="Weight: CLIP")
    ap.add_argument("--w_pose", type=float, default=0.3, help="Weight: Pose")
    ap.add_argument("--w_color", type=float, default=0.1, help="Weight: Color")
    ap.add_argument("--require_pd", action="store_true", help="Require Public Domain/CC0")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) load model + index
    model, preprocess, device = load_clip()
    index, ids, meta = load_index(Path(args.index_dir))
    matcher = Matcher(index, ids, meta, model, preprocess, device=device)

    # 2) read input image
    img = Image.open(args.img).convert("RGB")

    # 3) search
    weights = dict(w_clip=args.w_clip, w_pose=args.w_pose, w_color=args.w_color)
    filters = {"require_public_domain": bool(args.require_pd)}
    results = matcher.search(img, k=args.k, weights=weights, filters=filters, topn=args.topk)

    # 4) print results
    if not results:
        print("[RESULTS] no match found (try relaxing filters or using another image).")
        return

    print("\n[RESULTS]")
    for i, (rid, m, score) in enumerate(results, 1):
        title  = m.get("artwork_title_en") or m.get("title") or "Untitled"
        artist = m.get("artist_name_en")   or m.get("artistDisplayName") or "Unknown"
        year   = m.get("year")             or m.get("objectDate") or "?"
        lic    = m.get("license") or ("Public Domain" if m.get("isPublicDomain") else "?")
        path   = m.get("image_path") or "-"
        print(f"{i:02d}. {title} — {artist} ({year}) | License: {lic} | Score={score:.3f} | {path}")

    # 5) optionally show first match (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import cv2

        top_path = results[0][1].get("image_path")
        if top_path and Path(top_path).exists():
            ref_bgr = cv2.imread(top_path)
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

            fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

            ax0 = plt.subplot(gs[0]); ax0.imshow(img); ax0.set_title("Input"); ax0.axis("off")
            ax1 = plt.subplot(gs[1]); ax1.imshow(ref_rgb); ax1.set_title("Top Match"); ax1.axis("off")

            plt.tight_layout(); plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
