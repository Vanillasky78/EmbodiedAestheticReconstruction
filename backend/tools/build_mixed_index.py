#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge embeddings from data/local and data/met into a global mixed index.

Input:
  data/local/embeddings.npy
  data/local/embeddings_meta.csv
  data/met/embeddings.npy
  data/met/embeddings_meta.csv

Output:
  data/mixed/embeddings.npy
  data/mixed/embeddings_meta.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up to project root
DATA_ROOT = PROJECT_ROOT / "data"


def load_embeddings(museum: str):
    emb_path = DATA_ROOT / museum / "embeddings.npy"
    meta_path = DATA_ROOT / museum / "embeddings_meta.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"{museum}: embeddings not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"{museum}: meta CSV not found: {meta_path}")

    emb = np.load(emb_path)
    if isinstance(emb, np.lib.npyio.NpzFile):
        emb = emb["embeddings"]

    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    meta["museum"] = museum  # keep track of source
    return emb, meta


def main():
    local_emb, local_meta = load_embeddings("local")
    met_emb, met_meta = load_embeddings("met")

    # stack embeddings
    mixed_emb = np.concatenate([local_emb, met_emb], axis=0)
    print(f"[mixed] embeddings shape = {mixed_emb.shape}")

    mixed_dir = DATA_ROOT / "mixed"
    os.makedirs(mixed_dir, exist_ok=True)

    emb_out = mixed_dir / "embeddings.npy"
    meta_out = mixed_dir / "embeddings_meta.csv"

    np.save(emb_out, mixed_emb)

    # concat meta (keep filename first)
    mixed_meta = pd.concat([local_meta, met_meta], ignore_index=True)
    if "filename" in mixed_meta.columns:
        cols = ["filename"] + [c for c in mixed_meta.columns if c != "filename"]
        mixed_meta = mixed_meta[cols]

    mixed_meta.to_csv(meta_out, index=False, encoding="utf-8")
    print(f"[mixed] saved embeddings → {emb_out}")
    print(f"[mixed] saved meta       → {meta_out}")


if __name__ == "__main__":
    main()
