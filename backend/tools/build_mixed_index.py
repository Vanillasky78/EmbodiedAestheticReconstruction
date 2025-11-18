#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge embeddings from multiple museums into a single global "mixed" index.

Typical layout (per museum):

  data/
    local/
      embeddings.npy
      embeddings_meta.csv
    met/
      embeddings.npy
      embeddings_meta.csv
    aic/
      embeddings.npy
      embeddings_meta.csv
    mixed/
      embeddings.npy          # <- this script writes here
      embeddings_meta.csv

Usage examples:

  # Merge local + met (old behavior)
  python backend/tools/build_mixed_index.py --museums local met --data_root data --out_museum mixed

  # Merge local + met + aic
  python backend/tools/build_mixed_index.py --museums local met aic --data_root data --out_museum mixed
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_one_museum(
    museum: str,
    data_root: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and metadata for a single museum.

    Returns:
        (embeddings, meta_df)

    Raises:
        FileNotFoundError if required files are missing.
        AssertionError if shapes do not match.
    """
    base = data_root / museum
    emb_path = base / "embeddings.npy"
    meta_path = base / "embeddings_meta.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"[{museum}] embeddings not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"[{museum}] embeddings_meta not found: {meta_path}")

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)

    if len(meta) != emb.shape[0]:
        raise AssertionError(
            f"[{museum}] row mismatch: embeddings={emb.shape[0]} vs meta={len(meta)}"
        )

    # Ensure we know which museum this row came from
    if "museum" not in meta.columns:
        meta["museum"] = museum
    else:
        # Fill empty values with museum id if needed
        meta["museum"] = meta["museum"].fillna(museum)

    print(
        f"[{museum}] loaded embeddings={emb.shape}  meta_rows={len(meta)} "
        f"from {emb_path.name}, {meta_path.name}"
    )
    return emb, meta


def build_mixed_index(
    museums: List[str],
    data_root: str = "data",
    out_museum: str = "mixed",
) -> None:
    """
    Merge embeddings/meta from given museums into data/<out_museum>/.
    """
    data_root_path = Path(data_root)
    out_base = data_root_path / out_museum
    out_base.mkdir(parents=True, exist_ok=True)

    all_embs: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []

    for m in museums:
        try:
            emb, meta = load_one_museum(m, data_root_path)
        except Exception as exc:
            print(f"[WARN] Skip museum '{m}' due to error: {exc}")
            continue

        all_embs.append(emb)
        all_meta.append(meta)

    if not all_embs:
        print("[ERROR] No museum loaded successfully. Nothing to merge.")
        return

    # Concatenate along rows
    mixed_emb = np.concatenate(all_embs, axis=0)
    mixed_meta = pd.concat(all_meta, ignore_index=True)

    # Put filename first (if present), then museum, then others
    cols = list(mixed_meta.columns)
    ordered_cols: List[str] = []

    for col in ["filename", "file", "image_path"]:
        if col in cols:
            ordered_cols.append(col)

    if "museum" in cols:
        ordered_cols.append("museum")

    # Add remaining columns
    for c in cols:
        if c not in ordered_cols:
            ordered_cols.append(c)

    mixed_meta = mixed_meta[ordered_cols]

    # Save
    emb_out = out_base / "embeddings.npy"
    meta_out = out_base / "embeddings_meta.csv"

    np.save(emb_out, mixed_emb)
    mixed_meta.to_csv(meta_out, index=False, encoding="utf-8")

    print(f"[{out_museum}] embeddings shape = {mixed_emb.shape}")
    print(f"[{out_museum}] saved embeddings → {emb_out}")
    print(f"[{out_museum}] saved meta       → {meta_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple museum indices.")
    parser.add_argument(
        "--museums",
        nargs="+",
        required=True,
        help="List of museum folder names under data/, e.g. local met aic",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data folder that contains each museum subfolder.",
    )
    parser.add_argument(
        "--out_museum",
        type=str,
        default="mixed",
        help="Name of the merged folder under data/ (default: mixed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "------------------------------------------------------------\n"
        f"[INFO] Building mixed index from museums={args.museums} "
        f"→ out='{args.out_museum}' (data_root={args.data_root})"
    )
    build_mixed_index(
        museums=args.museums,
        data_root=args.data_root,
        out_museum=args.out_museum,
    )
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
