# 用模块模式运行：python -m indexing.build_index --corpus data/corpus_sample/paintings --outdir data/indices/paintings
import os
import json
import faiss
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# 保证能 import 到 processors
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from processors.pose import extract_pose
from processors.retrieval import load_clip, embed_image

def iter_images(root_dir):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="图像语料目录")
    ap.add_argument("--outdir", required=True, help="索引输出目录")
    ap.add_argument("--max_images", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model, preprocess, device = load_clip()

    feats = []
    paths = []
    poses = []

    imgs = list(iter_images(args.corpus))
    if args.max_images > 0:
        imgs = imgs[:args.max_images]

    for path in tqdm(imgs, desc="Embedding & Pose"):
        try:
            img = Image.open(path).convert("RGB")
            f = embed_image(model, preprocess, device, img)[0]
            feats.append(f)
            pres = extract_pose(img)
            poses.append(pres.keypoints if pres.ok else None)
            paths.append({"path": path})
        except Exception as e:
            print("skip:", path, e)

    feats = np.stack(feats).astype("float32")
    faiss.normalize_L2(feats)
    index = faiss.index_factory(feats.shape[1], "IDMap,Flat")
    index.add_with_ids(feats, np.arange(len(feats)).astype("int64"))

    np.save(os.path.join(args.outdir, "feats.f32.npy"), feats)
    np.save(os.path.join(args.outdir, "poses.npy"), np.array(poses, dtype=object))
    with open(os.path.join(args.outdir, "paths.jsonl"), "w", encoding="utf-8") as f:
        for rec in paths:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    faiss.write_index(index, os.path.join(args.outdir, "faiss.index"))
    print(f"[done] saved index to {args.outdir} with {len(paths)} images")

if __name__ == "__main__":
    main()
