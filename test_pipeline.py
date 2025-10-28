"""
test_pipeline.py
----------------------------------------
Lightweight sanity check for the Embodied Aesthetic Reconstruction system.

This script does NOT use the camera.
Instead, it:
1. Loads a pre-generated skeleton pose image (black background, white joints/limbs).
2. Encodes that pose image using OpenCLIP (ViT-B/32).
3. Loads the curated portrait artwork database and its embeddings.
4. Computes cosine similarity and prints the Top-3 matches.

Use this to verify:
- open_clip is installed and working
- embeddings in data/embeddings/ are readable
- retrieval logic works
- CSV and file paths are correct

Run:
    python test_pipeline.py --pose data/interim/pose_frames/example_pose.png
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from open_clip import create_model_and_transforms

from processors.retrieval import ArtworkDatabase


CSV_PATH = "data/portrait_works.csv"
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def encode_pose_image(pose_img_path, clip_model, preprocess, device):
    """
    Load a pose skeleton image (RGB), convert to CLIP embedding.
    Returns np.ndarray shape (1, D), L2-normalized.
    """
    img = Image.open(pose_img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb /= emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy()  # shape (1, D)


def retrieve_top_k(user_emb, db, k=3):
    """
    Manual version of the retrieval stage.
    We do this explicitly in test_pipeline for clarity.
    """
    # Load all artwork embeddings + rows
    embeddings_matrix, rows = db.load_all_embeddings()  # embeddings_matrix: (N, D)

    # Compute cosine similarity
    scores = cosine_similarity(user_emb, embeddings_matrix)[0]  # shape (N,)

    # Get top-k indices
    idx_sorted = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(idx_sorted, start=1):
        row = rows[idx]
        result = {
            "rank": rank,
            "artist": row["artist_en"],
            "title": row["title_en"],
            "year": row["year"],
            "score": float(scores[idx]),
            "notes_pose": row["notes_pose"],
            "file_name": row["file_name"],
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose",
        required=True,
        help="Path to a skeleton pose image (e.g. data/interim/pose_frames/example_pose.png)"
    )
    args = parser.parse_args()

    pose_img_path = args.pose
    if not os.path.exists(pose_img_path):
        raise FileNotFoundError(f"Pose image not found: {pose_img_path}")

    print(f"🔧 Device: {DEVICE}")
    print(f"🔍 Using pose image: {pose_img_path}")

    # 1. Init CLIP
    print("📦 Loading OpenCLIP model...")
    clip_model, preprocess, _ = create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    # 2. Encode the provided pose skeleton
    print("🧍 Encoding pose skeleton...")
    user_emb = encode_pose_image(
        pose_img_path,
        clip_model,
        preprocess,
        DEVICE
    )  # shape (1, D)

    # 3. Load portrait database
    print("📚 Loading artwork database...")
    db = ArtworkDatabase(csv_path=CSV_PATH)

    # 4. Retrieve top matches
    print("🎯 Retrieving Top-3 matches...")
    results = retrieve_top_k(user_emb, db, k=3)

    # 5. Print results to console
    print("\n[RESULTS]")
    for item in results:
        print(f"{item['rank']}. {item['artist']} — {item['title']} ({item['year']})")
        print(f"   Similarity score: {item['score']:.4f}")
        print(f"   Pose reading: {item['notes_pose']}")
        print(f"   Image file: data/images/{item['file_name']}")
        print()

    print("✅ Test pipeline completed successfully.")


if __name__ == "__main__":
    main()
