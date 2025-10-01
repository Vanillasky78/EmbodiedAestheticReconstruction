"""
Batch retrieval test: run retrieval on all images in a folder
and save Top-K results to CSV.

Usage:
    python batch_test.py --folder examples/inputs --out results.csv --topk 5
"""

import argparse
import os
import csv
from PIL import Image

from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores
from processors.pose import extract_pose, pose_similarity


def iter_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Input folder of images")
    ap.add_argument("--index", default="data/indices/paintings", help="Index directory")
    ap.add_argument("--out", default="results.csv", help="CSV output file")
    ap.add_argument("--topk", type=int, default=5, help="Top-K retrievals per input")
    ap.add_argument("--alpha", type=float, default=0.8, help="Fusion weight (clip vs pose)")
    args = ap.parse_args()

    # 1. Load CLIP + Index
    model, preprocess, device = load_clip()
    idx = CorpusIndex(args.index)

    # 2. Prepare CSV
    fout = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    writer.writerow(["input_image", "rank", "match_path", "clip_score", "pose_score", "fused_score"])

    # 3. Iterate images
    for img_path in iter_images(args.folder):
        print(f"[proc] {img_path}")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("skip (bad image):", img_path, e)
            continue

        qfeat = embed_image(model, preprocess, device, img)
        qpose = extract_pose(img)
        qpose_kps = qpose.keypoints if qpose.ok else None

        sims, ids = idx.search(qfeat, topk=args.topk * 3)
        results = []
        for s, i in zip(sims, ids):
            path = idx.paths[i]
            ps = None
            if qpose_kps is not None:
                cpose = idx.poses[i]
                if cpose is not None:
                    ps = pose_similarity(qpose_kps, cpose)
            final = fuse_scores(float(s), ps, args.alpha)
            results.append((path, float(s), ps, final))

        results.sort(key=lambda x: x[3], reverse=True)
        results = results[:args.topk]

        # 写入 CSV
        for rank, (path, cs, ps, fs) in enumerate(results, start=1):
            writer.writerow([img_path, rank, path, f"{cs:.4f}", f"{ps:.4f}" if ps else "", f"{fs:.4f}"])

    fout.close()
    print(f"[done] Results saved to {args.out}")


if __name__ == "__main__":
    main()
