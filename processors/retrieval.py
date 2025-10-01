import os
import json
import faiss
import numpy as np
from typing import List, Optional
from PIL import Image

import open_clip, torch

def load_clip(device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess, device

@torch.no_grad()
def embed_image(model, preprocess, device, img: Image.Image) -> np.ndarray:
    x = preprocess(img).unsqueeze(0).to(device)
    f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype("float32")

class CorpusIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        req = ["faiss.index", "feats.f32.npy", "paths.jsonl", "poses.npy"]
        missing = [f for f in req if not os.path.exists(os.path.join(index_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"索引目录不完整：{index_dir}\n缺少：{', '.join(missing)}\n"
                f"请先运行：python -m indexing.build_index --corpus data/corpus_sample/paintings --outdir {index_dir}"
            )
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.feats = np.load(os.path.join(index_dir, "feats.f32.npy"))
        self.paths: List[str] = []
        with open(os.path.join(index_dir, "paths.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.paths.append(json.loads(line)["path"])
        self.poses = np.load(os.path.join(index_dir, "poses.npy"), allow_pickle=True)  # array of (N,2) or None

    def search(self, qfeat: np.ndarray, topk=5):
        sims, ids = self.index.search(qfeat, topk)
        return sims[0], ids[0]

def fuse_scores(clip_sim: float, pose_sim: Optional[float], alpha: float = 0.8) -> float:
    if pose_sim is None:
        return float(clip_sim)
    return float(alpha * clip_sim + (1 - alpha) * pose_sim)
