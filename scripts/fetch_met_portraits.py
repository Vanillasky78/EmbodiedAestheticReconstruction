# scripts/fetch_met_portraits.py

"""
Fetch a small curated portrait dataset from The Met Museum Open Access API.
"""

import csv, time, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "met"
IMG_DIR = OUT_DIR / "images"
OUT_CSV = OUT_DIR / "portrait_works.csv"

SEARCH_TERMS = [
    "portrait", "woman portrait", "man portrait",
    "lady portrait", "self-portrait",
]

MAX_RESULTS = 80

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def met_search_ids(query: str):
    url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    params = {"q": query, "hasImages": "true", "isOnView": "true"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return []
    ids = r.json().get("objectIDs") or []
    return ids[:100]


def fetch_object(object_id: int):
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    d = r.json()

    img = d.get("primaryImage") or d.get("primaryImageSmall")
    if not img:
        return None

    if d.get("isPublicDomain") is not True:
        return None

    return {
        "object_id": object_id,
        "title": d.get("title", ""),
        "artist": d.get("artistDisplayName", "Unknown"),
        "year": d.get("objectDate", ""),
        "image_url": img,
        "museum": "Metropolitan Museum of Art, New York",
        "permalink": d.get("objectURL", ""),
    }


def download_image(url: str, path: Path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return True
    except:
        return False
    return False


def main():
    print("=== Fetching Met Museum Portraits ===")

    all_ids = []
    for q in SEARCH_TERMS:
        ids = met_search_ids(q)
        print(f"{q} → {len(ids)} ids")
        all_ids.extend(ids)
        time.sleep(0.5)

    all_ids = list(dict.fromkeys(all_ids))  # dedupe

    rows = []
    count = 0

    for oid in all_ids:
        if count >= MAX_RESULTS:
            break

        data = fetch_object(oid)
        if not data:
            continue

        filename = f"{oid}.jpg"
        img_path = IMG_DIR / filename

        if not download_image(data["image_url"], img_path):
            continue

        rows.append({
            "filename": filename,
            "title": data["title"],
            "artist": data["artist"],
            "year": data["year"],
            "museum": data["museum"],
            "license": "CC0",
            "price_text": "",
            "auction_price_usd": "",
            "permalink": data["permalink"],
        })

        count += 1
        print(f"[{count}] Saved {filename}")
        time.sleep(0.2)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Saved {count} images.")
    print(f"Images → {IMG_DIR}")
    print(f"CSV    → {OUT_CSV}")

if __name__ == "__main__":
    main()
