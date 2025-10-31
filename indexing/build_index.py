# indexing/build_index.py
"""
Build a FAISS index from your portrait dataset JSONL + images.

Usage:
  conda activate ear-mvp
  python indexing/build_index.py \
    --dataset_jsonl data/interim/portrait_art_dataset.jsonl \
    --images_dir data/images \
    --out_dir indexing \
    --device auto \
    --model ViT-B-32 \
    --pretrained openai \
    --batch_size 16 \
    --verbose
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import faiss
from PIL import Image
import open_clip


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True, help="Path to JSONL with rows (must include 'file_name').")
    ap.add_argument("--images_dir", required=True, help="Directory with image files.")
    ap.add_argument("--out_dir", required=True, help="Directory to write faiss.index / ids.npy / meta.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--model", default="ViT-B-32", help="open_clip model name")
    ap.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def pick_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad JSON on line {i}: {e} -> skipped")
    return rows


def resolve_image_path(row: Dict, images_dir: str) -> Optional[str]:
    """
    Resolve an image file path for a row.
    Priority:
      1) file_name (exact)
      2) file_name with trimmed spaces (already trimmed outside, but keep)
      3) derive from embedding_path basename + common extensions
    """
    # 1) From file_name
    fname = (row.get("file_name") or "").strip()
    if fname:
        p = os.path.join(images_dir, fname)
        if os.path.isfile(p):
            return p

    # 2) Try from embedding_path base name
    epath = (row.get("embedding_path") or "").strip()
    if epath:
        base = os.path.splitext(os.path.basename(epath))[0]
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            cand = os.path.join(images_dir, base + ext)
            if os.path.isfile(cand):
                return cand

    return None


@torch.no_grad()
def encode_images(
    rows: List[Dict],
    images_dir: str,
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 16,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Returns:
        embs: (N, D) float32
        ids:  (N,) int64
        meta: list of row metadata aligned with embs
    """
    paths: List[str] = []
    kept_rows: List[Dict] = []

    # Resolve paths with robust rules
    for r in rows:
        # strip string fields to avoid hidden spaces
        for k, v in list(r.items()):
            if isinstance(v, str):
                r[k] = v.strip()
        img_path = resolve_image_path(r, images_dir)
        if img_path is None:
            if verbose:
                print(f"[WARN] missing/invalid image for id={r.get('id')} file_name='{r.get('file_name')}' -> skipped")
            continue
        paths.append(img_path)
        kept_rows.append(r)

    if len(paths) == 0:
        raise RuntimeError("No embeddings created. Make sure images exist under --images_dir and JSONL has valid 'file_name'.")

    if verbose:
        print(f"[info] encoding {len(paths)} images on device={device}")

    model = model.to(device)
    model.eval()

    all_embs: List[torch.Tensor] = []
    all_ids: List[int] = []

    # some rows may not have numeric id; fall back to index position
    def safe_id(r: Dict, i: int) -> int:
        try:
            return int(r.get("id"))
        except Exception:
            return i

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                if verbose:
                    print(f"[WARN] failed to open/preprocess {p}: {e} -> skipped")
                images.append(None)

        # filter out Nones but keep alignment for ids/meta
        valid_idx = [j for j, im in enumerate(images) if im is not None]
        if not valid_idx:
            continue

        imgs = torch.stack([images[j] for j in valid_idx], dim=0).to(device)
        feats = model.encode_image(imgs)
        feats = feats.float()
        # Normalize for cosine/inner-product FAISS
        feats = torch.nn.functional.normalize(feats, dim=1)

        all_embs.append(feats.cpu())
        for j in valid_idx:
            row_idx = i + j
            all_ids.append(safe_id(kept_rows[row_idx], row_idx))

    if not all_embs:
        raise RuntimeError("No embeddings created after preprocessing. Check image files and formats.")

    embs = torch.cat(all_embs, dim=0).numpy().astype("float32")
    ids = np.array(all_ids, dtype=np.int64)

    # meta aligned with kept & successfully processed rows
    meta: List[Dict] = []
    cursor = 0
    for k in range(0, len(paths), batch_size):
        end = min(k + batch_size, len(paths))
        # we don't know exactly which inside each batch failed at PIL step,
        # so rebuild meta from length of feats per batch (already handled via valid_idx)
        # simpler: recompute valid_idx again for deterministic alignment
        valid_idx = list(range(end - k))  # all assumed valid at this stage
        # But we filtered above; so rely on ids length:
        # We'll append meta entries to match len(ids) order
        # To keep it simple, push meta for the slice we actually encoded using ids mapping:
        pass

    # A simpler, correct alignment: rebuild meta by walking all_ids and mapping back to row with same "safe id".
    id_to_row = {}
    for idx, r in enumerate(kept_rows):
        rid = r.get("id")
        try:
            rid = int(rid)
        except Exception:
            rid = idx
        id_to_row.setdefault(rid, r)

    meta = [id_to_row[rid] for rid in all_ids]

    return embs, ids, meta


def build_faiss(embs: np.ndarray) -> faiss.Index:
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (works with normalized vectors as cosine)
    index.add(embs)
    return index


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device(args.device)
    if args.verbose:
        print(f"[device] {device}")
        # quick sanity for data lines
        try:
            n = sum(1 for _ in open(args.dataset_jsonl, "r", encoding="utf-8"))
            print(f"[data] rows: {n}")
        except Exception:
            pass

    rows = load_jsonl(args.dataset_jsonl)
    if args.verbose:
        print(f"[jsonl] parsed rows: {len(rows)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )

    embs, ids, meta = encode_images(
        rows=rows,
        images_dir=args.images_dir,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"[embeddings] shape={embs.shape}, ids={ids.shape}")

    index = build_faiss(embs)

    faiss_path = os.path.join(args.out_dir, "faiss.index")
    ids_path = os.path.join(args.out_dir, "ids.npy")
    meta_path = os.path.join(args.out_dir, "meta.json")

    faiss.write_index(index, faiss_path)
    np.save(ids_path, ids)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote:\n  {faiss_path}\n  {ids_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
