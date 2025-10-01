"""
Quick pipeline test: load CLIP + FAISS + Pose, run retrieval on one image
Usage:
    python test_pipeline.py --img path/to/input.jpg
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image

from processors.retrieval import load_clip, embed_image, CorpusIndex, fuse_scores
from processors.pose import extract_pose, pose_similarity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Input image for testing")
    ap.add_argument("--index", default="data/indices/paintings", help="Index directory")
    ap.add_argument("--topk", type=int, default=5, help="Top-K to retrieve")
    ap.add_argument("--alpha", type=float, default=0.8, help="Fusion weight (clip vs pose)")
    args = ap.parse_args()

    # 1. Load model + index
    model, preprocess, device = load_clip()
    idx = CorpusIndex(args.index)

    # 2. Read input image
    img = Image.open(args.img).convert("RGB")

    # 3. Embed & Pose
    qfeat = embed_image(model, preprocess, device, img)
    qpose = extract_pose(img)
    qpose_kps = qpose.keypoints if qpose.ok else None

    # 4. Retrieval
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

    # 5. Print results
    print("\n[RESULTS]")
    for path, cs, ps, fs in results:
        print(f"{path} | CLIP={cs:.3f} | Pose={ps:.3f} | Fused={fs:.3f}")

    # 6. Optional: show first match visually
    try:
        import matplotlib.pyplot as plt
        ref = cv2.imread(results[0][0])
        inp = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)); plt.title("Input")
        plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)); plt.title("Top Match")
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
