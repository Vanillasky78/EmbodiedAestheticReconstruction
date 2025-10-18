# app.py — Match-Only UI for Embodied Aesthetic Reconstruction
# ------------------------------------------------------------
# Usage:
#   streamlit run app.py
#
# Prereqs (build index first):
#   python indexing/build_index.py \
#     --dataset_jsonl data/interim/portrait_art_dataset.jsonl \
#     --images_dir    data/images

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import streamlit as st
from PIL import Image

# ML + ANN
import torch
import open_clip
import faiss

# our retrieval logic
from processors.retrieval import Matcher


# --------------------------
# App / page config
# --------------------------
st.set_page_config(page_title="Portrait Match (Match-Only)", layout="wide")

# --------------------------
# Paths & constants
# --------------------------
INDEX_DIR  = Path("indexing")
INDEX_PATH = INDEX_DIR / "faiss.index"
IDS_PATH   = INDEX_DIR / "ids.npy"
META_PATH  = INDEX_DIR / "meta.json"


# --------------------------
# Lazy-load CLIP + FAISS
# --------------------------
@st.cache_resource(show_spinner=True)
def load_clip_and_index() -> Tuple[Matcher | None, Any | None, Any | None, str | None]:
    """Load CLIP model + FAISS index + metadata, return a ready Matcher."""
    # 1) CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)

    # 2) Index files
    missing = [p for p in (INDEX_PATH, IDS_PATH, META_PATH) if not p.exists()]
    if missing:
        return None, None, None, device

    try:
        index = faiss.read_index(str(INDEX_PATH))
        ids   = np.load(str(IDS_PATH), allow_pickle=True)
        meta  = json.loads(META_PATH.read_text(encoding="utf-8"))

        if not (len(ids) == len(meta)):
            st.warning(f"Index/meta length mismatch: ids={len(ids)} meta={len(meta)}")
        matcher = Matcher(index, ids, meta, model, preprocess, device=device)
        return matcher, model, preprocess, device
    except Exception as e:
        st.error(f"Failed to load index/meta: {e}")
        return None, None, None, device


matcher, model, preprocess, device = load_clip_and_index()

# If index missing, guide the user and stop.
if matcher is None:
    st.title("Portrait Match (Match-Only)")
    st.error(
        "Index not found.\n\n"
        "Please build it first:\n\n"
        "```bash\n"
        "python indexing/build_index.py \\\n"
        "  --dataset_jsonl data/interim/portrait_art_dataset.jsonl \\\n"
        "  --images_dir    data/images\n"
        "```\n"
        "Then rerun: `streamlit run app.py`"
    )
    st.stop()

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("Match-Only Settings")
w_clip  = st.sidebar.slider("Weight: CLIP",  0.0, 1.0, 0.60, 0.05)
w_pose  = st.sidebar.slider("Weight: Pose",  0.0, 1.0, 0.30, 0.05)
w_color = st.sidebar.slider("Weight: Color", 0.0, 1.0, 0.10, 0.05)
require_pd = st.sidebar.checkbox("Require Public Domain/CC0", value=True)
TOPK = st.sidebar.slider("Top-K", 5, 50, 12)

st.title("Portrait Match (Match-Only)")
st.caption("Upload a person photo; the system returns the most similar portrait artworks (no fusion).")

# --------------------------
# File uploader
# --------------------------
img_file = st.file_uploader("Upload portrait photo", type=["jpg", "jpeg", "png"])
if img_file:
    try:
        q = Image.open(img_file).convert("RGB")
    except Exception:
        st.error("Failed to read the uploaded image. Please try another file.")
        st.stop()

    st.image(q, caption="Input", use_column_width=True)

    with st.spinner("Searching best matches..."):
        weights = dict(w_clip=w_clip, w_pose=w_pose, w_color=w_color)
        results = matcher.search(
            q, k=200, weights=weights,
            filters={"require_public_domain": require_pd},
            topn=TOPK,
        )

    if not results:
        st.warning("No results. Try disabling the Public Domain filter or using a clearer portrait image.")
    else:
        # grid display
        cols = st.columns(min(4, TOPK))
        for i, (rid, meta, score) in enumerate(results):
            c = cols[i % len(cols)]
            with c:
                title  = meta.get("artwork_title_en") or meta.get("title") or "Untitled"
                artist = meta.get("artist_name_en") or meta.get("artistDisplayName") or "Unknown"
                year   = meta.get("year") or meta.get("objectDate") or "?"
                museum = meta.get("museum") or meta.get("department") or ""
                lic    = meta.get("license") or ("Public Domain" if meta.get("isPublicDomain") else "?")
                img_p  = meta.get("image_path")

                st.markdown(f"**{title}**")
                st.caption(f"{artist} • {year} • {museum}")
                st.caption(f"License: {lic} | Score: {score:.3f}")
                if img_p and Path(img_p).exists():
                    st.image(img_p, use_column_width=True)
                else:
                    st.info("Image not cached locally.")

# Footer
st.markdown("---")
st.caption("MSc Final Project — Embodied Aesthetic Reconstruction | Match-Only Mode (no fusion).")
