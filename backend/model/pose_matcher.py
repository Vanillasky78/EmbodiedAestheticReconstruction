# model/pose_matcher.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
from model.utils import load_clip_model, preprocess_image, cosine_similarity

class PoseMatcher:
    def __init__(self, data_root, clip_model_name, clip_pretrained, topk=3,
                 use_dual=False, allow_museums=None, device=None, verbose=False):
        self.data_root = data_root
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.topk = topk
        self.use_dual = use_dual
        self.verbose = verbose
        self.device = device
        self.allow_museums = allow_museums

        self.model, self.preprocess = load_clip_model(clip_model_name, clip_pretrained, device)
        self.embeddings = {}
        self.metadatas = {}

        self._load_all()

    def _load_all(self):
        roots = [d for d in os.listdir(self.data_root)
                 if os.path.isdir(os.path.join(self.data_root, d)) and not d.startswith(".")]
        for museum in roots:
            if self.allow_museums and museum not in self.allow_museums:
                continue

            emb_path = os.path.join(self.data_root, museum, "embeddings.npy")
            meta_path = os.path.join(self.data_root, museum, "embeddings_meta.csv")
            if not os.path.exists(emb_path) or not os.path.exists(meta_path):
                continue

            try:
                self.embeddings[museum] = np.load(emb_path)
                self.metadatas[museum] = pd.read_csv(meta_path)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed loading {museum}: {e}")

    def match(self, image: Image.Image, museum: str = "local", topk: int = 3):
        if museum not in self.embeddings:
            return {"error": f"Museum '{museum}' not indexed."}

        image_input = preprocess_image(image, self.preprocess)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input.to(self.model.device)).cpu().numpy()

        sims = cosine_similarity(image_features, self.embeddings[museum])
        top_indices = np.argsort(sims)[::-1][:topk]

        results = []
        for idx in top_indices:
            row = self.metadatas[museum].iloc[idx].to_dict()
            row["score"] = float(sims[idx])
            results.append(row)

        return results

    def get_metadata(self, museum: str, filename: str):
        if museum not in self.metadatas:
            return {}
        df = self.metadatas[museum]
        row = df[df.filename == filename].to_dict("records")
        return row[0] if row else {}

    def get_museum_list(self):
        return list(self.embeddings.keys())