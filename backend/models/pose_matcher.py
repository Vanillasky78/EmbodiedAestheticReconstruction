# backend/pose_matcher.py
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from model.utils import load_clip_model, compute_image_embedding, dual_crop
from config import Settings
import json

# Optional: open API imports
from open_apis import match_met, search_europeana

class PoseMatcher:
    def __init__(
        self,
        data_root: str,
        model_name: str,
        pretrained: str,
        device: Optional[str] = None,
        dual_view: bool = True,
        verbose: bool = False,
    ):
        self.data_root = data_root
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.dual_view = dual_view
        self.verbose = verbose
        self.model, self.preprocess = load_clip_model(model_name, pretrained, device=device)
        self.db = {}

        self._load_all_museums()

    def _load_all_museums(self):
        for museum in os.listdir(self.data_root):
            folder = os.path.join(self.data_root, museum)
            if os.path.isdir(folder):
                self.db[museum] = self._load_museum(museum)

    def _load_museum(self, museum: str):
        embed_path = os.path.join(self.data_root, museum, "embeddings.npy")
        meta_path = os.path.join(self.data_root, museum, "embeddings_meta.csv")
        if not os.path.exists(embed_path):
            return None
        embeddings = np.load(embed_path)
        metas = []
        with open(meta_path, "r", encoding="utf-8") as f:
            headers = f.readline().strip().split(",")
            for line in f:
                values = line.strip().split(",")
                metas.append(dict(zip(headers, values)))
        return {"embeddings": embeddings, "metas": metas}

    def match(self, image: Image.Image, museum: str = "local", topk: int = 3) -> List[Dict]:
        image_input = dual_crop(image) if self.dual_view else [self.preprocess(image)]
        image_embed = compute_image_embedding(self.model, image_input, self.device)

        if museum == "met":
            return match_met(image_embed, topk=topk)
        elif museum == "europeana":
            return search_europeana(image_embed, topk=topk)
        elif museum == "mixed":
            all_results = []
            for m in self.db.keys():
                all_results += self._match_single(image_embed, m, topk=topk)
            all_results.sort(key=lambda x: -x["score"])
            return all_results[:topk]
        else:
            return self._match_single(image_embed, museum, topk=topk)

    def _match_single(self, image_embed, museum: str, topk: int = 3):
        if museum not in self.db:
            return []
        db = self.db[museum]
        sims = image_embed @ db["embeddings"].T
        scores = sims.max(axis=0)
        topk_idx = np.argsort(-scores)[:topk]
        results = []
        for idx in topk_idx:
            meta = db["metas"][idx]
            meta = {k: (v if v != "" else None) for k, v in meta.items()}
            meta.update({"score": float(scores[idx]), "museum": museum})
            results.append(meta)
        return results