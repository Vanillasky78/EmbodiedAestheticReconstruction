# indexing/build_index.py
# ------------------------------------------------------------
# Build CLIP feature index for a local portrait art dataset
# Example:
#   python indexing/build_index.py \
#       --dataset_jsonl data/interim/portrait_art_dataset.jsonl \
#       --images_dir data/images \
#       --out_dir indexing \
#       --batch_size 32 --device auto --clip_model ViT-B-32 --clip_pretrained openai
# ------------------------------------------------------------

from __future__ import annotations
import json, argparse, random
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ----------------------------
# Dependencies
# ----------------------------
try:
    import open_clip, torch
    from PIL import Image
except Exception:
    raise SystemExit("Install required packages: pip install open-clip-torch torch pillow")

try:
    import faiss
except Exception:
    raise SystemExit("Install required packages: pip install faiss-cpu")

# ----------------------------
# Utilities
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
BASE_DIR = Path(__file__).parent.resolve()


def load_model(
    device: str | None = None,
    arch: str = "ViT-B-32",
    pretrained: str = "openai",
):
    """
    Load CLIP model and preprocessing transforms.
    device: 'cpu' | 'cuda' | 'auto' | None
    """
    dev = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval().to(dev)
    return model, preprocess, dev


def embed_batch(paths, model, preprocess, device, batch_size=32):
    """
    Encode images in batches. Unreadable images are dropped.
    Returns:
        (embeddings: (N, D) float32, kept_paths: List[str])
    """
    import torch
    all_feats, keep_paths = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
            batch = paths[i : i + batch_size]
            imgs, ok = [], []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))
                    ok.append(True)
                except Exception:
                    ok.append(False)
            if not imgs:
                continue
            t = torch.stack(imgs, dim=0).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            all_feats.append(f.cpu().numpy().astype("float32"))
            for j, good in enumerate(ok):
                if good:
                    keep_paths.append(batch[j])

    if not all_feats:
        return np.zeros((0, 512), dtype="float32"), []
    return np.vstack(all_feats), keep_paths


def find_local_image(record: dict, images_dir: Path) -> str | None:
    """
    Find a local image file that matches an artwork record.
    Priority:
      1) record['image_path'] if exists
      2) glob by artist+title fragments
      3) glob by source prefix
    """
    # 1) Provided absolute/relative path
    ipath = record.get("image_path")
    if ipath and Path(ipath).exists():
        return str(ipath)

    # 2) artist + title fragments
    artist = str(record.get("artist_name_en", "")).strip().replace(" ", "_")
    title = str(record.get("artwork_title_en", "")).strip().replace(" ", "_")
    pat1 = [
        str(p) for p in images_dir.glob(f"*{artist[:15]}*{title[:15]}*")
        if p.suffix.lower() in EXTS
    ]
    if pat1:
        return pat1[0]

    # 3) fallback by source prefix
    src = str(record.get("source_api", "")).strip()
    pat2 = [str(p) for p in images_dir.glob(f"{src}_*") if p.suffix.lower() in EXTS]
    return pat2[0] if pat2 else None


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from local portrait art dataset.")
    parser.add_argument("--dataset_jsonl", required=True, help="Path to dataset .jsonl file")
    parser.add_argument("--images_dir", required=True, help="Directory with local image files")
    parser.add_argument("--out_dir", default=str(BASE_DIR), help="Output folder for index and metadata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="'cpu' | 'cuda' | 'auto'")
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="openai")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    lines = Path(args.dataset_jsonl).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]

    images_dir = Path(args.images_dir)
    candidates, metas, ids = [], [], []

    # Map JSONL entries to local images
    for rec in rows:
        p = find_local_image(rec, images_dir)
        if not p:
            continue
        candidates.append(p)
        ids.append(rec.get("artwork_id"))
        metas.append({
            "artwork_id": rec.get("artwork_id"),
            "artist_name_en": rec.get("artist_name_en"),
            "artwork_title_en": rec.get("artwork_title_en"),
            "year": rec.get("year"),
            "license": rec.get("license"),
            "museum": rec.get("museum"),
            "image_path": p,
        })

    # Deduplicate paths (keep last) and sort for reproducibility
    if candidates:
        uniq = {}
        for p, i, m in zip(candidates, ids, metas):
            uniq[p] = (i, m)
        candidates = sorted(uniq.keys())
        ids, metas = zip(*(uniq[p] for p in candidates))
        ids, metas = list(ids), list(metas)

    if not candidates:
        raise SystemExit("No images matched. Please run the dataset downloader or check --images_dir path.")

    model, preprocess, device = load_model(args.device, args.clip_model, args.clip_pretrained)
    embeddings, good_paths = embed_batch(candidates, model, preprocess, device, batch_size=args.batch_size)

    # Synchronize and remove failed entries if any
    if len(good_paths) != len(candidates):
        keep = set(good_paths)
        kept_ids, kept_meta = [], []
        for p, i, m in zip(candidates, ids, metas):
            if p in keep:
                kept_ids.append(i)
                kept_meta.append(m)
        ids, metas = kept_ids, kept_meta

    # Save outputs
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "ids.npy", np.array(ids, dtype=object))
    (out_dir / "meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    print(f"✅ Saved index with {len(ids)} items → {out_dir}")


if __name__ == "__main__":
    main()
