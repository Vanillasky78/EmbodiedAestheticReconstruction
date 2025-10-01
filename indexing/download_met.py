# 用模块模式运行：python -m indexing.download_met --count 50 --resume
import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from PIL import Image
from tqdm import tqdm

SEARCH_API = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

UA = "EAR-Retrieval/1.0 (+for research; contact: you@example.com)"  # 可自定义
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

def ensure_dir(p: Path):
    if p.exists() and p.is_file():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)

def search_ids(q: str = "painting", department_id: Optional[int] = None,
               public_domain: bool = True, has_images: bool = True) -> List[int]:
    params = {"q": q}
    if public_domain:
        params["isPublicDomain"] = "true"
    if has_images:
        params["hasImages"] = "true"
    if department_id is not None:
        params["departmentId"] = str(department_id)
    r = SESSION.get(SEARCH_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("objectIDs", []) or []

def iter_objects(ids: Iterable[int], retry: int = 3, sleep: float = 0.15):
    for oid in ids:
        for attempt in range(1, retry + 1):
            try:
                r = SESSION.get(OBJECT_API.format(oid), timeout=30)
                r.raise_for_status()
                yield oid, r.json()
                break
            except Exception as e:
                if attempt == retry:
                    print(f"[warn] object {oid} failed: {e}")
                else:
                    time.sleep(sleep * attempt)

def pick_image_url(info: dict, prefer_original: bool = False) -> Optional[str]:
    url = info.get("primaryImage") if prefer_original else info.get("primaryImageSmall")
    if not url:
        url = info.get("primaryImage") or info.get("primaryImageSmall")
    return url or None

def download_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    img.load()
    return img.convert("RGB")

def main():
    ap = argparse.ArgumentParser(description="Download Met Museum Open Access images.")
    ap.add_argument("--q", default="painting", help="Search query (default: painting)")
    ap.add_argument("--count", type=int, default=50, help="How many to save")
    ap.add_argument("--outdir", default="data/corpus_sample/paintings", help="Output folder")
    ap.add_argument("--department", type=int, default=None, help="Department ID filter")
    ap.add_argument("--prefer_original", action="store_true", help="Prefer original over small")
    ap.add_argument("--min_w", type=int, default=400, help="Min width to accept")
    ap.add_argument("--min_h", type=int, default=400, help="Min height to accept")
    ap.add_argument("--resume", action="store_true", help="Skip files already on disk")
    ap.add_argument("--sleep", type=float, default=0.15, help="Delay between requests")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    meta_path = outdir / "metadata.jsonl"

    print("[info] querying object IDs…")
    ids = search_ids(q=args.q, department_id=args.department)
    if not ids:
        print("[error] no results from Met API. try another query.")
        return

    existing = set()
    if args.resume:
        for p in outdir.glob("*.jpg"):
            try:
                existing.add(int(p.stem))
            except ValueError:
                pass

    saved = 0
    with open(meta_path, "a", encoding="utf-8") as mf:
        for oid, info in tqdm(iter_objects(ids), total=len(ids), desc="objects"):
            if args.resume and oid in existing:
                continue

            img_url = pick_image_url(info, prefer_original=args.prefer_original)
            if not img_url:
                continue

            meta = {
                "objectID": oid,
                "title": info.get("title"),
                "artistDisplayName": info.get("artistDisplayName"),
                "objectDate": info.get("objectDate"),
                "classification": info.get("classification"),
                "department": info.get("department"),
                "culture": info.get("culture"),
                "objectURL": info.get("objectURL"),
                "image": img_url,
            }

            try:
                img = download_image(img_url)
                w, h = img.size
                if w < args.min_w or h < args.min_h:
                    continue
                fname = outdir / f"{oid}.jpg"
                img.save(fname, quality=90)

                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                saved += 1
                time.sleep(args.sleep)
                if saved >= args.count:
                    break
            except Exception as e:
                print(f"[warn] {oid} image failed: {e}")
                continue

    print(f"[done] saved {saved} jpgs → {outdir}")
    print(f"[meta] appended provenance to {meta_path}")

if __name__ == "__main__":
    main()
