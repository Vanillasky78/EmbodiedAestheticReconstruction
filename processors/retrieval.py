# processors/retrieval.py
import os, json, faiss, numpy as np
from PIL import Image

import open_clip, torch

def load_clip(device="cuda" if torch.cuda.is_available() else "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    return model.to(device).eval(), preprocess, device

@torch.no_grad()
def embed_image(model, preprocess, device, img: Image.Image):
    x = preprocess(img).unsqueeze(0).to(device)
    f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype("float32")

class CorpusIndex:
    def __init__(self, index_dir: str):
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.feats = np.load(os.path.join(index_dir, "feats.f32.npy"))
        with open(os.path.join(index_dir, "paths.jsonl"), "r", encoding="utf-8") as f:
            self.paths = [json.loads(line)["path"] for line in f]
        self.poses = np.load(os.path.join(index_dir, "poses.npy"), allow_pickle=True)

    def search(self, qfeat: np.ndarray, topk=5):
        sims, ids = self.index.search(qfeat, topk)
        return sims[0], ids[0]

def fuse_scores(clip_sim: float, pose_sim: float|None, alpha: float=0.8) -> float:
    return alpha*clip_sim + (1-alpha)*(pose_sim if pose_sim else 0)
