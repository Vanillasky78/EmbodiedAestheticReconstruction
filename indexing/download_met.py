# indexing/download_met.py
# ------------------------------------------------------------
# Lightweight sampler for The Met Open Access collection.
#
# Usage (module mode):
#   python -m indexing.download_met --q "portrait,self-portrait" --count 80 --resume
#   python -m indexing.download_met --q "portrait" --prefer_original --min_w 800 --min_h 800 --count 60
#
# Output:
#   data/corpus_sample/paintings/xxxx.jpg
#   data/corpus_sample/paintings/metadata.jsonl  (+ optional metadata.csv)
#
# Notes:
# - For your main dataset, prefer the dedicated portrait-art-dataset pipeline/notebook.
# ------------------------------------------------------------

from __future__ import annotations
import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ----------------------------
# API endpoints & session
# ----------------------------
SEARCH_API = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

UA = "EAR-Retrieval/1.0 (+for research; contact: you@example.com)"  # customize if needed
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# ----------------------------
# Tunables
# ----------------------------
HTTP_TIMEOUT = 30
SEARCH_RETRY = 3
OBJECT_RETRY = 3
IMAGE_RETRY = 3

POLITE_DELAY = 0.15        # seconds between object fetches
RETRY_BACKOFF = 0.25       # additional seconds * attempt

MIN_JPEG_QUALITY = 90
ACCEPT_CT_PREFIXES = ("image/",)  # content-type must start with these


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    """Create directory (remove file with same name if exists)."""
    if p.exists() and p.is_file():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


def _get_json(url: str, params: Optional[Dict[str, Any]] = None, retry: int = 3) -> Dict[str, Any]:
    """GET JSON with basic retry."""
    for attempt in range(1, retry + 1):
        try:
            r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retry:
                raise
            time.sleep(RETRY_BACKOFF * attempt)


def search_ids(
    q: str = "painting",
    department_id: Optional[int] = None,
    public_domain: bool = True,
    has_images: bool = True,
) -> List[int]:
    """Query objectIDs for a single keyword."""
    params: Dict[str, str] = {"q": q}
    if public_domain:
        params["isPublicDomain"] = "true"
    if has_images:
        params["hasImages"] = "true"
    if department_id is not None:
        params["departmentId"] = str(department_id)

    data = _get_json(SEARCH_API, params=params, retry=SEARCH_RETRY)
    return data.get("objectIDs", []) or []


def search_ids_multi(
    queries: List[str],
    department_id: Optional[int] = None,
    public_domain: bool = True,
    has_images: bool = True,
) -> List[int]:
    """Union IDs from multiple keywords; return sorted unique list for reproducibility."""
    all_ids = set()
    for q in queries:
        q = q.strip()
        if not q:
            continue
        try:
            ids = search_ids(q=q, department_id=department_id, public_domain=public_domain, has_images=has_images)
            all_ids.update(ids)
        except Exception as e:
            print(f"[warn] search for '{q}' failed: {e}")
    return sorted(all_ids)


def iter_objects(ids: Iterable[int], retry: int = OBJECT_RETRY, sleep: float = POLITE_DELAY):
    """Yield (object_id, object_json) with retry and polite delay."""
    for oid in ids:
        for attempt in range(1, retry + 1):
            try:
                data = _get_json(OBJECT_API.format(oid), retry=1)  # _get_json already wraps retry
                yield oid, data
                break
            except Exception as e:
                if attempt == retry:
                    print(f"[warn] object {oid} failed: {e}")
                else:
                    time.sleep(RETRY_BACKOFF * attempt)
        time.sleep(sleep)


def pick_image_url(info: dict, prefer_original: bool = False) -> Optional[str]:
    """
    Choose the best image URL.
    Prefer small for speed unless prefer_original=True.
    """
    url = info.get("primaryImage") if prefer_original else info.get("primaryImageSmall")
    if not url:
        url = info.get("primaryImage") or info.get("primaryImageSmall")
    return url or None


def _is_image_response(r: requests.Response) -> bool:
    ct = r.headers.get("Content-Type", "")
    return any(ct.startswith(pfx) for pfx in ACCEPT_CT_PREFIXES)


def download_image(url: str, retry: int = IMAGE_RETRY) -> Optional[Image.Image]:
    """Download and decode an image, with content-type check and retry."""
    for attempt in range(1, retry + 1):
        try:
            r = SESSION.get(url, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            if not _is_image_response(r):
                raise ValueError(f"unexpected content-type: {r.headers.get('Content-Type')}")
            img = Image.open(BytesIO(r.content))
            img.load()
            return img.convert("RGB")
        except (UnidentifiedImageError, Exception) as e:
            if attempt == retry:
                print(f"[warn] image GET failed ({url}): {e}")
                return None
            time.sleep(RETRY_BACKOFF * attempt)


def build_meta(oid: int, info: dict, img_url: str) -> dict:
    """Normalize a subset of useful metadata fields."""
    return {
        "objectID": oid,
        "title": info.get("title"),
        "artistDisplayName": info.get("artistDisplayName"),
        "artistDisplayBio": info.get("artistDisplayBio"),
        "objectDate": info.get("objectDate"),
        "classification": info.get("classification"),
        "department": info.get("department"),
        "culture": info.get("culture"),
        "objectURL": info.get("objectURL"),
        "primaryImage": info.get("primaryImage"),
        "primaryImageSmall": info.get("primaryImageSmall"),
        "isPublicDomain": info.get("isPublicDomain"),
        "image": img_url,
    }


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Download The Met Open Access images (portraits sampler).")
    ap.add_argument("--q", default="portrait,self-portrait",
                    help='Comma-separated search keywords. Default: "portrait,self-portrait"')
    ap.add_argument("--count", type=int, default=50, help="Max number of images to save")
    ap.add_argument("--outdir", default="data/corpus_sample/paintings", help="Output folder")
    ap.add_argument("--department", type=int, default=None, help="Department ID filter")
    ap.add_argument("--prefer_original", action="store_true", help="Prefer original over small")
    ap.add_argument("--min_w", type=int, default=400, help="Min width to accept")
    ap.add_argument("--min_h", type=int, default=400, help="Min height to accept")
    ap.add_argument("--resume", action="store_true", help="Skip files already on disk")
    ap.add_argument("--sleep", type=float, default=POLITE_DELAY, help="Delay between requests (politeness)")
    ap.add_argument("--csv", action="store_true", help="Also write metadata.csv (optional)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    meta_path = outdir / "metadata.jsonl"

    # 1) Query IDs
    print("[info] querying object IDs…")
    queries = [s.strip() for s in args.q.split(",") if s.strip()]
    ids = search_ids_multi(queries, department_id=args.department)
    if not ids:
        print("[error] no results from Met API. Try another query.")
        return

    # 2) Resume support
    existing = set()
    if args.resume:
        for p in outdir.glob("*.jpg"):
            try:
                existing.add(int(p.stem))
            except ValueError:
                pass

    # 3) Iterate objects, download images + write metadata
    saved = 0
    with open(meta_path, "a", encoding="utf-8") as mf:
        for oid, info in tqdm(iter_objects(ids, sleep=args.sleep), total=len(ids), desc="objects"):
            if args.resume and oid in existing:
                continue

            img_url = pick_image_url(info, prefer_original=args.prefer_original)
            if not img_url:
                continue

            img = download_image(img_url)
            if img is None:
                continue

            w, h = img.size
            if w < args.min_w or h < args.min_h:
                continue

            # Save image & metadata
            fname = outdir / f"{oid}.jpg"
            try:
                img.save(fname, quality=MIN_JPEG_QUALITY)
                meta = build_meta(oid, info, img_url)
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                saved += 1
            except Exception as e:
                print(f"[warn] saving {oid} failed: {e}")
                continue

            if saved >= args.count:
                break

    print(f"[done] saved {saved} images → {outdir}")
    print(f"[meta] appended → {meta_path}")

    # 4) Optional CSV export
    if args.csv:
        try:
            import pandas as pd  # optional
            rows = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines()]
            if rows:
                pd.DataFrame(rows).to_csv(outdir / "metadata.csv", index=False)
                print(f"[meta] also wrote CSV → {outdir / 'metadata.csv'}")
        except Exception as e:
            print(f"[warn] CSV export failed: {e}")


if __name__ == "__main__":
    main()
