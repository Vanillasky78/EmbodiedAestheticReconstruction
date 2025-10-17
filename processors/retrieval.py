from __future__ import annotations
from . import pose as pose_mod
from .similarity import pose_similarity, color_similarity, combine_score
import cv2, numpy as np

class Matcher:
    def __init__(self, faiss_index, id_list, meta_list, clip_model, clip_preprocess, device="cpu"):
        self.index = faiss_index
        self.ids = id_list
        self.meta = meta_list
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device

    def encode_image(self, pil_img):
        import torch
        with torch.no_grad():
            t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            return f.cpu().numpy().astype("float32")

    def search(self, pil_img, k=200, weights=None, filters=None):
        # Step 1: CLIP+FAISS shortlist
        q = self.encode_image(pil_img)
        sims, idxs = self.index.search(q, k)
        idxs = idxs[0]
        base = [(int(i), float(s)) for i, s in zip(idxs, sims[0]) if i >= 0]

        # Step 2: Pose & color for the query
        pr = pose_mod.extract_pose(pil_img)
        q_kps = pr.keypoints
        q_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

        # Step 3: Re-rank by multi-metric
        ranked = []
        for i, s_clip in base:
            m = self.meta[i]
            if filters and filters.get('require_public_domain', False):
                lic = str(m.get('license','')).lower()
                if lic not in ['public domain','cc0']:
                    continue
            cand = cv2.imread(m.get('image_path','')) if m.get('image_path') else None
            sim_pose = 0.0  # placeholder: add candidate pose cache if available
            sim_color = color_similarity(q_bgr, cand) if cand is not None else 0.0
            bonus = 0.02 if str(m.get('license','')).lower() in ['public domain','cc0'] else 0.0
            score = combine_score(s_clip, sim_pose, sim_color, meta_bonus=bonus, w=weights)
            ranked.append((i, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [(self.ids[i], self.meta[i], sc) for i, sc in ranked[:50]]