#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapter for the Art Institute of Chicago (AIC) collection CSV.

It reads the expanded artworks CSV, filters to works that have images,
optionally restricts to portrait-like works, and produces a unified
`embeddings_meta.csv` plus downloads images into a local folder.

Expected input layout (you can change paths via CLI flags):

  data/aic/raw/artic_allArtworks_expanded.csv
  data/aic/images/                      # will be created if missing
  data/aic/embeddings_meta.csv          # output of this script

Usage example:

  python backend/tools/adapter_aic.py \
      --csv data/aic/raw/artic_allArtworks_expanded.csv \
      --images-dir data/aic/images \
      --out-meta data/aic/embeddings_meta.csv \
      --max-images 800

You can re-run with a larger --max-images later if you want more data.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm


# -------------- basic helpers --------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def lower_join(*vals: Iterable[str]) -> str:
    parts: List[str] = []
    for v in vals:
        if isinstance(v, (list, tuple, set)):
            parts.extend([safe_str(x).lower() for x in v])
        else:
            parts.append(safe_str(v).lower())
    return " | ".join(parts)


# -------------- portrait & value heuristics --------------


PORTRAIT_KEYWORDS = [
    "portrait",
    "self-portrait",
    "self portrait",
    "selfportrait",
    "bust",
    "head of",
    "buste",
    "head study",
    "profile of",
    "mrs.",
    "mr.",
    "lady",
    "woman",
    "girl",
    "boy",
    "man",
    "figure",
]


def is_portrait_like(row: Dict[str, str]) -> bool:
    """
    Very simple heuristic: look for portrait-ish words in title,
    artwork_type_title, classification_titles, category_titles, term_titles, etc.
    """
    title = row.get("title", "")
    art_type = row.get("artwork_type_title", "")
    dept = row.get("department_title", "")
    cats = row.get("category_titles", "")
    terms = row.get("term_titles", "")
    styles = row.get("style_titles", "")
    classif = row.get("classification_titles", "")

    text = lower_join(
        title,
        art_type,
        dept,
        cats,
        terms,
        styles,
        classif,
    )

    return any(kw in text for kw in PORTRAIT_KEYWORDS)


def compute_value_score(row: Dict[str, str], tier: int, portrait_flag: int) -> float:
    """
    Tiny heuristic for 'value_score' in [0, 1]:
    - base on museum tier
    - boost for portrait-like works
    - boost for oil paintings and impressive credit lines
    """
    # start from tier-based baseline
    if tier == 1:
        base = 0.85
    elif tier == 2:
        base = 0.70
    else:
        base = 0.60

    if portrait_flag:
        base += 0.10

    medium = safe_str(row.get("medium_display", "")).lower()
    if "oil" in medium and "canvas" in medium:
        base += 0.05

    credit = safe_str(row.get("credit_line", "")).lower()
    if "gift of" in credit or "bequest of" in credit or "collection" in credit:
        base += 0.03

    # clamp to [0, 1]
    return float(max(0.0, min(1.0, base)))


# -------------- main adapter logic --------------


def iter_aic_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    """
    Stream rows from the big AIC CSV.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def build_image_url(image_id: str) -> str:
    """
    Build IIIF image URL for AIC.

    See: https://api.artic.edu/docs/#iiif-image-api
    """
    image_id = image_id.strip()
    return f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"


def download_image(url: str, out_path: Path, timeout: int = 30) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return False
        out_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def build_aic_adapter(
    csv_path: Path,
    images_dir: Path,
    out_meta: Path,
    max_images: Optional[int] = None,
    only_portraits: bool = True,
    tier: int = 2,
    skip_download: bool = False,
) -> None:
    ensure_dir(images_dir)
    ensure_dir(out_meta.parent)

    rows_meta: List[Dict[str, str]] = []

    total_kept = 0
    total_seen = 0

    for row in tqdm(iter_aic_rows(csv_path), desc="Scanning AIC CSV"):
        total_seen += 1

        img_id = safe_str(row.get("image_id", "")).strip()
        if not img_id:
            # no image => skip
            continue

        # decide if portrait or not
        portrait = 1 if is_portrait_like(row) else 0
        if only_portraits and not portrait:
            continue

        # respect max_images
        if max_images is not None and total_kept >= max_images:
            break

        # build local filename & download URL
        art_id = safe_str(row.get("id", "")).strip()
        if not art_id:
            # should not happen but just in case
            art_id = f"aic_{total_seen}"

        filename = f"aic_{art_id}.jpg"
        img_out_path = images_dir / filename
        image_url = build_image_url(img_id)

        # download if needed
        if not skip_download:
            if img_out_path.exists():
                ok = True
            else:
                ok = download_image(image_url, img_out_path)
            if not ok:
                # skip this entry if we cannot get the image
                continue

        # compact year from date_start / date_display
        year = ""
        date_start = safe_str(row.get("date_start", "")).strip()
        if date_start:
            year = date_start
        else:
            year = safe_str(row.get("date_display", "")).strip()

        artist = safe_str(row.get("artist_title", "")).strip()
        title = safe_str(row.get("title", "")).strip()
        dept = safe_str(row.get("department_title", "")).strip()
        classif = safe_str(row.get("classification_titles", "")).strip()
        style = safe_str(row.get("style_titles", "")).strip()
        terms = safe_str(row.get("term_titles", "")).strip()
        cats = safe_str(row.get("category_titles", "")).strip()
        credit = safe_str(row.get("credit_line", "")).strip()

        value_score = compute_value_score(row, tier=tier, portrait_flag=portrait)

        # construct a simple source URL from api_link if present
        api_link = safe_str(row.get("api_link", "")).strip()
        source_url = api_link or ""

        meta_row = {
            "id": art_id,
            "filename": filename,
            "title": title,
            "artist": artist,
            "year": year,
            "museum": "aic",
            "tier": str(tier),
            "portrait_flag": str(portrait),
            "masterpiece_flag": "0",  # you can manually flip some rows to 1 later
            "value_score": f"{value_score:.3f}",
            "image_url": image_url,
            "source_url": source_url,
            "credit_line": credit,
            "department": dept,
            "classification": classif,
            "style": style,
            "terms": terms,
            "categories": cats,
            "license": "unknown",  # you can refine later if needed
        }

        rows_meta.append(meta_row)
        total_kept += 1

    if not rows_meta:
        print("[WARN] No rows kept for AIC (maybe filters too strict?).")
        return

    # Write embeddings_meta.csv with a stable column order
    fieldnames = [
        "id",
        "filename",
        "title",
        "artist",
        "year",
        "museum",
        "tier",
        "portrait_flag",
        "masterpiece_flag",
        "value_score",
        "image_url",
        "source_url",
        "credit_line",
        "department",
        "classification",
        "style",
        "terms",
        "categories",
        "license",
    ]

    with out_meta.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_meta:
            writer.writerow(r)

    print(
        f"[OK] AIC adapter finished: kept={total_kept} / seen={total_seen}, "
        f"meta={out_meta}, images_dir={images_dir}"
    )


# -------------- CLI entrypoint --------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AIC embeddings_meta.csv and download images.")
    p.add_argument(
        "--csv",
        type=str,
        default="data/aic/raw/artic_allArtworks_expanded.csv",
        help="Path to AIC expanded artworks CSV.",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default="data/aic/images",
        help="Folder where images will be stored.",
    )
    p.add_argument(
        "--out-meta",
        type=str,
        default="data/aic/embeddings_meta.csv",
        help="Path to output embeddings_meta.csv.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=800,
        help="Max number of artworks to keep (None for all).",
    )
    p.add_argument(
        "--only-portraits",
        action="store_true",
        default=True,
        help="Keep only portrait-like works (default: True).",
    )
    p.add_argument(
        "--all-works",
        action="store_true",
        help="If set, ignore portrait filter and keep all works.",
    )
    p.add_argument(
        "--tier",
        type=int,
        default=2,
        help="Tier level for this museum (1=top, 2=high, 3=normal).",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="If set, do not download images, only build meta (filenames will still be generated).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    out_meta = Path(args.out_meta)

    if not csv_path.exists():
        raise SystemExit(f"[ERR] CSV not found: {csv_path}")

    only_portraits = not args.all_works  # default True, override with --all-works

    max_images = args.max_images
    if max_images is not None and max_images <= 0:
        max_images = None

    print(f"[INFO] Using CSV: {csv_path}")
    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Out meta: {out_meta}")
    print(f"[INFO] Only portraits: {only_portraits}")
    print(f"[INFO] Max images: {max_images}")
    print(f"[INFO] Tier: {args.tier}")
    print(f"[INFO] Skip download: {args.skip_download}")

    build_aic_adapter(
        csv_path=csv_path,
        images_dir=images_dir,
        out_meta=out_meta,
        max_images=max_images,
        only_portraits=only_portraits,
        tier=args.tier,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
