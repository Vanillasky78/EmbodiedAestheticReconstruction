"""
Fetch a small curated portrait dataset from The Met Museum Open Access API.

This script:
1. Searches for portraits / female portrait / male portrait
2. Downloads the first ~80 high-quality, CC0 images
3. Saves them into data/met/images/
4. Writes a metadata CSV: data/met/portrait_works.csv

You only need to run this once.

Usage:
    python scripts/fetch_met_portraits.py
"""

import csv
import os
from pathlib import Path
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "met"
IMG_DIR = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "portrait_works.csv"

SEARCH_TERMS = [
    "portrait",
    "woman portrait",
    "man portrait",
    "lady portrait",
    "self-portrait",
]

MAX_RESULTS = 80  # total images you want

# Ensure folders exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def met_search_ids(query: str):
    """Search Met API and return list of objectIDs."""
    url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    params = {
        "q": query,
        "hasImages": "true",
        "isOnView": "true"
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return []
    data = r.json()
    ids = data.get("objectIDs") or []
    return ids[:100]  # get up to 100 per keyword


def fetch_object(object_id: int):
    """Get object details + primaryImage."""
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()

    img = data.get("primaryImage") or data.get("primaryImageSmall")
    if not img:
        return None

    # Only accept CC0
    if data.get("isPublicDomain") != True:
        return None

    title = data.get("title") or ""
    artist = data.get("artistDisplayName") or "Unknown"
    year = data.get("objectDate") or ""
    museum = "Metropolitan Museum of Art, New York"
    permalink = data.get("objectURL") or ""

    return {
        "object_id": object_id,
        "title": title,
        "artist": artist,
        "year": year,
        "image_url": img,
        "museum": museum,
        "permalink": permalink,
    }


def download_image(url: str, out_path: Path):
    """Save image to disk."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        return False
    return False


def main():
    print("=== Fetching Met Museum portraits ===")

    all_ids = []
    for q in SEARCH_TERMS:
        ids = met_search_ids(q)
        print(f"{q} → {len(ids)} ids")
        all_ids.extend(ids)
        time.sleep(0.5)

    all_ids = list(dict.fromkeys(all_ids))  # remove duplicates
    print(f"Total unique IDs: {len(all_ids)}")

    rows = []
    count = 0

    for oid in all_ids:
        if count >= MAX_RESULTS:
            break

        try:
            item = fetch_object(oid)
        except Exception:
            item = None

        if not item:
            continue

        filename = f"{item['object_id']}.jpg"
        out_img = IMG_DIR / filename

        ok = download_image(item["image_url"], out_img)
        if not ok:
            continue

        row = {
            "filename": filename,
            "title": item["title"],
            "artist": item["artist"],
            "year": item["year"],
            "museum": item["museum"],
            "license": "CC0",
            "price_text": "",
            "auction_price_usd": "",
            "permalink": item["permalink"],
        }
        rows.append(row)
        count += 1
        print(f"[{count}] Saved {filename}")

        time.sleep(0.2)

    # Write metadata CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "title",
                "artist",
                "year",
                "museum",
                "license",
                "price_text",
                "auction_price_usd",
                "permalink",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Saved {count} images.")
    print(f"Images → {IMG_DIR}")
    print(f"CSV    → {OUT_CSV}")


if __name__ == "__main__":
    main()
