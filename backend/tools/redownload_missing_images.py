#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redownload missing images for AIC or any museum.
Checks embeddings_meta.csv for filenames,
and downloads those missing in images/.
"""

import csv
import os
import requests
from pathlib import Path
from tqdm import tqdm


def download(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False


def main():
    museum = "aic"
    data_root = Path("data") / museum

    csv_path = data_root / "embeddings_meta.csv"
    img_dir = data_root / "images"

    if not csv_path.exists():
        print(f"[ERR] Missing CSV: {csv_path}")
        return
    if not img_dir.exists():
        print(f"[ERR] Missing images dir: {img_dir}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Loaded {len(rows)} rows from {csv_path}")

    fixed = 0
    failed = 0

    for r in tqdm(rows, desc="Checking"):
        filename = r.get("filename", "").strip()
        url = r.get("image_url", "").strip()

        if not filename:
            continue

        out_path = img_dir / filename
        if out_path.exists():
            continue  # OK

        if not url:
            print(f"[WARN] No URL for {filename}")
            failed += 1
            continue

        # try download
        ok = download(url, out_path)
        if ok:
            fixed += 1
        else:
            print(f"[ERR] Download failed: {filename}  URL={url}")
            failed += 1

    print("--------------------------------------------------")
    print(f"[DONE] Fixed images: {fixed}")
    print(f"[DONE] Still missing: {failed}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
