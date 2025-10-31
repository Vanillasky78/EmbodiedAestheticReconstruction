# indexing/build_index.py
import os, json, argparse, sys, math, time
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image

try:
    import faiss                   # pip install faiss-cpu
except Exception as e:
    print("[ERR] faiss not available, please `pip install faiss-cpu`")
    raise

import open_clip                  # pip install open-clip-torch


def load_rows(jsonl_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@torch.no_grad()
def encode_images(rows: List[Dict[str, Any]],
                  images_dir: str,
                  model,
                  preprocess,
                  device: str,
                  save_pt_to: str = None) -> tuple:
    """
    Returns:
        (embs: np.ndarray [N, D], ids: np.ndarray [N], meta: List[Dict]])
    Also optionally saves per-image embeddings to path provided by each row['embedding_path'].
    """
    model.eval()

    all_embs: List[np.ndarray] = []
    ids: List[int] = []
    meta: List[Dict[str, Any]] = []

    # base dir for saving per-image embeddings (so that "embeddings/xxx.pt" is under data/)
    data_root = os.path.abspath(os.path.join(images_dir, os.pardir))

    for idx, row in enumerate(rows, 1):
        # robust get of file name
        fname = row.get("file_name") or row.get("image") or row.get("path") or row.get("filename")
        if not fname:
            print(f"[WARN] line {idx}: no 'file_name' (or alias) key in row -> skipped")
            continue

        img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(img_path):
            print(f"[WARN] missing image: {img_path} -> skipped")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open: {img_path} -> {e}")
            continue

        image = preprocess(image).unsqueeze(0).to(device)

        feat = model.encode_image(image)
        # normalize to unit length (Inner Product index = cosine if normalized)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        emb_np = feat.cpu().numpy().astype("float32")  # faiss expects float32

        # collect
        all_embs.append(emb_np[0])
        # id field fallback
        _id = row.get("id")
        if _id is None:
            _id = len(ids) + 1
        ids.append(int(_id))

        # save per-image .pt if user provided embedding_path
        epath = row.get("embedding_path")
        if epath:
            out_pt = os.path.join(data_root, epath) if not os.path.isabs(epath) else epath
            os.makedirs(os.path.dirname(out_pt), exist_ok=True)
            try:
                torch.save(torch.from_numpy(emb_np), out_pt)
            except Exception as e:
                print(f"[WARN] failed to save embedding to {out_pt}: {e}")

        # meta to help UI
        meta.append({
            "id": int(_id),
            "file_name": fname,
            "artist_en": row.get("artist_en"),
            "title_en": row.get("title_en"),
            "notes_pose": row.get("notes_pose"),
            "year": row.get("year")
        })

    if len(all_embs) == 0:
        raise RuntimeError("No embeddings created. Make sure images exist under --images_dir and JSONL has valid 'file_name'.")

    embs = np.stack(all_embs, axis=0)
    ids = np.asarray(ids, dtype=np.int64)
    return embs, ids, meta


def build_faiss(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)             # inner-product
    faiss.normalize_L2(embs)                  # safety: ensure already normalized
    index.add(embs)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True, help="data/interim/portrait_art_dataset.jsonl")
    ap.add_argument("--images_dir", required=True, help="data/images")
    ap.add_argument("--out_dir", required=True, help="indexing")
    ap.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[device] {args.device}")

    if not os.path.isfile(args.dataset_jsonl):
        raise FileNotFoundError(f"{args.dataset_jsonl} not found")
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"{args.images_dir} not found")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_rows(args.dataset_jsonl)
    print(f"[data] rows: {len(rows)}")

    # load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(args.device)

    embs, ids, meta = encode_images(
        rows=rows,
        images_dir=args.images_dir,
        model=model,
        preprocess=preprocess,
        device=args.device,
    )

    if args.verbose:
        print(f"[embeddings] shape={embs.shape}, ids={ids.shape}")

    index = build_faiss(embs)

    # save artifacts
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
