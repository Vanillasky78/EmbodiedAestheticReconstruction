# model/pose_matcher.py
# -----------------------------------------------------------------------------
# Embodied Aesthetic Reconstruction — PoseMatcher
# -----------------------------------------------------------------------------
# Core logic for multi-museum artwork retrieval.
# Combines YOLOv8-Pose (for skeletons) and OpenCLIP embeddings for image similarity.
# -----------------------------------------------------------------------------

import os
import glob
import warnings
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from PIL import Image

from ultralytics import YOLO
import torch
import open_clip

from .utils import load_image_from_bytes, l2_normalize, cosine_similarity
from ..config import Settings


class PoseMatcher:
    """Main class handling YOLO-Pose + CLIP based matching."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # ---- Device selection ----
        if settings.device_override:
            self.device = torch.device(settings.device_override)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ---- YOLOv8 Pose model ----
        model_path = settings.pose_model_path
        self.pose = YOLO(model_path if os.path.exists(model_path) else "yolov8n-pose.pt")

        # ---- CLIP model ----
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained,
            device=self.device,
        )
        self.clip_model.eval()

        # ---- Load all museum datasets ----
        self.datasets = self._load_museum_datasets(settings.data_root, settings.allow_museums)
        if not self.datasets:
            warnings.warn("⚠️ No datasets found under data_root; API will run but matching will fail.")

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return current model/device/data status."""
        total = sum(len(ds["meta"]) for ds in self.datasets.values()) if self.datasets else 0
        return {
            "device": str(self.device),
            "pose_model": getattr(self.pose.model, "name", "yolov8n-pose"),
            "clip_model": f"{self.settings.clip_model_name}/{self.settings.clip_pretrained}",
            "loaded_museums": {k: len(v["meta"]) for k, v in self.datasets.items()},
            "total_items": total,
        }

    def list_museums(self) -> Dict[str, int]:
        """List all loaded museum datasets."""
        return {name: len(ds["meta"]) for name, ds in self.datasets.items()}

    def get_metadata(self, museum: str, filename: str) -> Optional[Dict[str, Any]]:
        """Return metadata row for a given museum + filename."""
        ds = self.datasets.get(museum)
        if not ds:
            return None
        df = ds["meta"]
        row = df.loc[df["filename"] == filename]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def match_image_bytes(self, image_bytes: bytes, museum: Optional[str] = None, topk: int = 3) -> List[Dict[str, Any]]:
        """Main function: given an image (bytes), return Top-K most similar artworks."""
        if not self.datasets:
            raise ValueError("No datasets loaded. Please build embeddings first.")

        img: Image.Image = load_image_from_bytes(image_bytes)

        # 1) Pose estimation (optional)
        try:
            yres = self.pose.predict(source=img, verbose=False, imgsz=640)
            keypoints = yres[0].keypoints.xy.cpu().numpy().tolist() if yres and yres[0].keypoints is not None else []
        except Exception:
            keypoints = []

        # 2) Encode image to CLIP embedding
        with torch.no_grad():
            t_full = self.preprocess(img).unsqueeze(0).to(self.device)
            feat_full = self.clip_model.encode_image(t_full)
            feat_full = feat_full / feat_full.norm(dim=-1, keepdim=True)
            feat_full = feat_full[0].detach().cpu().numpy()

            # Optional dual-view: center crop
            if self.settings.use_dual_view:
                w, h = img.size
                short = min(w, h)
                left = (w - short) // 2
                top = (h - short) // 2
                crop = img.crop((left, top, left + short, top + short)).resize((w, h))
                t_crop = self.preprocess(crop).unsqueeze(0).to(self.device)
                feat_crop = self.clip_model.encode_image(t_crop)
                feat_crop = feat_crop / feat_crop.norm(dim=-1, keepdim=True)
                feat_crop = feat_crop[0].detach().cpu().numpy()
                feat = (feat_full + feat_crop) / 2.0
            else:
                feat = feat_full

        query = l2_normalize(feat, axis=0)  # (D,)

        # 3) Compute cosine similarity per museum
        museums = [museum] if museum else list(self.datasets.keys())
        all_hits = []

        for m in museums:
            ds = self.datasets.get(m)
            if not ds:
                continue
            embs = ds["embeddings"]  # (N, D)
            sims = cosine_similarity(embs, query)  # (N,)
            idxs = np.argsort(-sims)[:max(1, topk)]
            for rank, i in enumerate(idxs):
                rec = ds["meta"].iloc[int(i)].to_dict()
                all_hits.append({
                    "museum": m,
                    "rank": rank + 1,
                    "similarity": float(sims[i]),
                    **rec,
                    "keypoints": keypoints if rank == 0 else None,
                })

        all_hits.sort(key=lambda x: x["similarity"], reverse=True)
        return all_hits[:max(1, topk)]

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _load_museum_datasets(self, base: str, allow: Optional[List[str]]):
        """Scan data/<museum>/ for embeddings and metadata files."""
        datasets = {}
        subdirs = sorted([p for p in glob.glob(os.path.join(base, "*/"))])
        for folder in subdirs:
            name = os.path.basename(os.path.dirname(folder))
            if allow and name not in allow:
                continue
            emb = os.path.join(folder, "embeddings.npy")
            meta = os.path.join(folder, "embeddings_meta.csv")
            if os.path.exists(emb) and os.path.exists(meta):
                embs = np.load(emb)
                embs = embs.astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
                embs = embs / norms  # ensure normalized
                datasets[name] = {
                    "embeddings": embs,
                    "meta": pd.read_csv(meta),
                }
        return datasets
