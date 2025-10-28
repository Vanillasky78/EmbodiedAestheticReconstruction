"""
generate_embeddings.py
----------------------------------------
This script generates an OpenCLIP embedding (.pt tensor file)
for each reference artwork in data/portrait_works.csv.

It:
1. Loads the curated portrait dataset CSV.
2. Loads each artwork image from data/images/<file_name>.
3. Encodes the image with OpenCLIP (ViT-B/32, pretrained='openai').
4. Saves the embedding as a .pt file into data/embeddings/.
5. Ensures the CSV has a valid embedding_path for each row.

Run this once before using:
    - test_pipeline.py
    - match_pose_to_artworks.py
    - app.py

Author: Xinyi Zhang
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_and_transforms


# ------------------------------
# Config
# ------------------------------

CSV_PATH = "data/portrait_works.csv"
IMAGES_DIR = "data/images"
EMBED_DIR = "data/embeddings"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_clip_model(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE):
    """
    Load OpenCLIP model and preprocessing transforms.
    """
    model, preprocess, _ = create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def encode_image_to_embedding(img_path, model, preprocess, device=DEVICE):
    """
    Given an image path, return a normalized embedding tensor (1, D).
    """
    img = Image.open(img_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_t)              # (1, D)
        emb = emb / emb.norm(dim=-1, keepdim=True)   # normalize

    return emb.cpu()  # keep on CPU for saving


def ensure_dirs():
    os.makedirs(EMBED_DIR, exist_ok=True)


def main():
    print(f"🔧 Device: {DEVICE}")
    print("📂 Loading CSV:", CSV_PATH)

    ensure_dirs()

    # load metadata
    df = pd.read_csv(CSV_PATH)

    # load model once
    print("🎛 Loading OpenCLIP model (ViT-B/32)...")
    clip_model, preprocess = load_clip_model()

    # we'll update embedding_path values as we go
    updated_embedding_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding artworks"):
        file_name = row["file_name"]  # e.g. "Ingres_Moitessier_Seated.jpg"
        raw_img_path = os.path.join(IMAGES_DIR, file_name)

        if not os.path.exists(raw_img_path):
            print(f"⚠️ WARNING: image not found for row {idx}: {raw_img_path}")
            updated_embedding_paths.append(row.get("embedding_path", ""))
            continue

        # Decide output .pt path
        # Prefer existing embedding_path column if it's valid and inside EMBED_DIR
        existing_path = str(row.get("embedding_path", "")).strip()
        if existing_path and existing_path.lower() != "nan":
            out_path = existing_path
        else:
            # default to data/embeddings/<basename>.pt
            base_no_ext = os.path.splitext(file_name)[0]
            out_path = os.path.join(EMBED_DIR, f"{base_no_ext}.pt")

        # Make sure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # If file already exists, we can skip recompute unless you want to refresh
        # We'll recompute anyway for consistency
        emb = encode_image_to_embedding(raw_img_path, clip_model, preprocess, DEVICE)

        # Save tensor
        torch.save(emb, out_path)

        updated_embedding_paths.append(out_path)

    # Update dataframe with the final embedding paths (so downstream code can trust it)
    df["embedding_path"] = updated_embedding_paths

    # Write CSV back
    df.to_csv(CSV_PATH, index=False)
    print("✅ Embeddings generated and CSV updated.")
    print(f"📄 Saved CSV: {CSV_PATH}")
    print(f"💾 Embeddings directory: {EMBED_DIR}")


if __name__ == "__main__":
    main()
