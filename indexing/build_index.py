from __future__ import annotations
import os, json, argparse, glob
import numpy as np
from tqdm import tqdm

try:
    import open_clip, torch
    from PIL import Image
except Exception:
    raise SystemExit("Install: pip install open-clip-torch torch pillow")

try:
    import faiss
except Exception:
    raise SystemExit("Install: pip install faiss-cpu")


def load_model(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    return model, preprocess, device


def embed_paths(paths, model, preprocess, device):
    embs = []
    with torch.no_grad():
        for p in tqdm(paths, desc="Embedding"):
            try:
                img = Image.open(p).convert("RGB")
                t = preprocess(img).unsqueeze(0).to(device)
                f = model.encode_image(t)
                f = f / f.norm(dim=-1, keepdim=True)
                embs.append(f.cpu().numpy())
            except Exception:
                embs.append(np.zeros((1,512), dtype=np.float32))
    return np.vstack(embs).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="indexing")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.dataset_jsonl, "r", encoding="utf-8").read().splitlines()]

    paths, ids, metas = [], [], []
    for r in rows:
        artist = str(r.get("artist_name_en", "")).strip().replace(" ", "_")
        title = str(r.get("artwork_title_en", "")).strip().replace(" ", "_")
        cand = glob.glob(os.path.join(args.images_dir, f"*{artist[:15]}*{title[:15]}*.jpg"))
        if not cand:
            src = r.get("source_api", "")
            cand = glob.glob(os.path.join(args.images_dir, f"{src}_*.jpg"))
        if not cand:
            continue
        p = cand[0]
        paths.append(p)
        ids.append(r.get("artwork_id"))
        metas.append({
            "artwork_id": r.get("artwork_id"),
            "artist_name_en": r.get("artist_name_en"),
            "artwork_title_en": r.get("artwork_title_en"),
            "year": r.get("year"),
            "license": r.get("license"),
            "museum": r.get("museum"),
            "image_path": p,
        })

    if not paths:
        raise SystemExit("No images mapped. Run the dataset downloader first.")

    model, preprocess, device = load_model()
    embs = embed_paths(paths, model, preprocess, device)

    np.save(os.path.join(args.out_dir, "embeddings.npy"), embs)
    np.save(os.path.join(args.out_dir, "ids.npy"), np.array(ids))
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))
    print(f"Saved index with {len(ids)} items → {args.out_dir}")

if __name__ == "__main__":
    main()
processors/similarity.py
from __future__ import annotations
import cv2, numpy as np
from typing import Dict

# Pose similarity (normalized Kx2 arrays)
def pose_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        m = min(a.shape[0], b.shape[0])
        a, b = a[:m], b[:m]
    def _norm(x):
        x = x - x.mean(0, keepdims=True)
        s = np.linalg.norm(x)
        return x / (s + 1e-6)
    a, b = _norm(a), _norm(b)
    d = np.linalg.norm(a - b) / np.sqrt(a.size)
    return float(max(0.0, 1.0 - d))

# Color histogram similarity (HSV Bhattacharyya)
def color_similarity(bgr_a: np.ndarray | None, bgr_b: np.ndarray | None) -> float:
    if bgr_a is None or bgr_b is None:
        return 0.0
    def _hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
        h = cv2.normalize(h,h).flatten()
        return h
    ha, hb = _hist(bgr_a), _hist(bgr_b)
    sim = cv2.compareHist(ha.astype('float32'), hb.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - sim))

# Weighted fusion
default_w = dict(w_clip=0.6, w_pose=0.3, w_color=0.1)

def combine_score(sim_clip: float, sim_pose: float, sim_color: float, meta_bonus: float = 0.0, w: Dict[str,float] = None) -> float:
    w = w or default_w
    s = (w.get('w_clip',0.6)*sim_clip + w.get('w_pose',0.3)*sim_pose + w.get('w_color',0.1)*sim_color + meta_bonus)
    return float(max(0.0, min(1.0, s)))