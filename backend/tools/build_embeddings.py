# tools/build_embeddings.py
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from model.utils import load_clip_model, preprocess_image
import torch
from tqdm import tqdm

def generate_embeddings(image_folder, model, preprocess):
    embeddings = []
    filenames = []
    for fname in tqdm(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        try:
            image = Image.open(os.path.join(image_folder, fname)).convert("RGB")
            image_tensor = preprocess_image(image, preprocess).to(model.device)
            with torch.no_grad():
                emb = model.encode_image(image_tensor).cpu().numpy()[0]
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[WARN] Failed on {fname}: {e}")

    return np.stack(embeddings), filenames

def save_embeddings(museum_dir, embeddings, filenames, meta_csv=None):
    np.save(os.path.join(museum_dir, "embeddings.npy"), embeddings)

    meta_df = pd.DataFrame({"filename": filenames})
    if meta_csv and os.path.exists(meta_csv):
        try:
            external = pd.read_csv(meta_csv)
            meta_df = meta_df.merge(external, on="filename", how="left")
        except Exception as e:
            print(f"[WARN] Failed to merge metadata: {e}")

    meta_df.to_csv(os.path.join(museum_dir, "embeddings_meta.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--museum", type=str, help="Name of museum folder under data/")
    parser.add_argument("--all", action="store_true", help="Process all museum folders")
    parser.add_argument("--meta", type=str, help="Optional CSV file with metadata")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from config import Settings
    settings = Settings()
    model, preprocess = load_clip_model(settings.clip_model_name, settings.clip_pretrained, settings.device_override)

    museum_dirs = []
    if args.all:
        museum_dirs = [d for d in os.listdir(settings.data_root)
                       if os.path.isdir(os.path.join(settings.data_root, d))]
    elif args.museum:
        museum_dirs = [args.museum]
    else:
        print("Please set --museum=NAME or use --all.")
        return

    for museum in museum_dirs:
        museum_path = os.path.join(settings.data_root, museum)
        image_folder = os.path.join(museum_path, "images")
        if not os.path.exists(image_folder):
            print(f"[ERROR] Folder not found: {image_folder}")
            continue

        print(f"[INFO] Processing '{museum}'...")
        embeddings, filenames = generate_embeddings(image_folder, model, preprocess)
        save_embeddings(museum_path, embeddings, filenames, args.meta)
        print(f"[OK] {museum}: saved {len(filenames)} embeddings.")

if __name__ == "__main__":
    main()
