# app.py — Match-Only UI for Embodied Aesthetic Reconstruction
# ------------------------------------------------------------
# Usage:
#   streamlit run app.py
# Prereqs:
#   python indexing/build_index.py --dataset_jsonl data/interim/portrait_art_dataset.jsonl --images_dir data/images

import os, json
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
INDEX_DIR = "indexing"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
IDS_PATH   = os.path.join(INDEX_DIR, "ids.npy")
META_PATH  = os.path.join(INDEX_DIR, "meta.json")

# --------------------------
# Lazy-load CLIP + FAISS
# --------------------------
@st.cache_resource(show_spinner=True)
def load_clip_and_index():
    # 1) CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)

    # 2) Index files
    if not (os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH) and os.path.exists(META_PATH)):
        return None, None, None, None

    index = faiss.read_index(INDEX_PATH)
    ids   = np.load(IDS_PATH, allow_pickle=True)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3) Build matcher
    matcher = Matcher(index, ids, meta, model, preprocess, device=device)
    return matcher, model, preprocess, device

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
    q = Image.open(img_file).convert("RGB")
    st.image(q, caption="Input", use_column_width=True)

    with st.spinner("Searching best matches..."):
        weights = dict(w_clip=w_clip, w_pose=w_pose, w_color=w_color)
        results = matcher.search(
            q, k=200, weights=weights,
            filters={"require_public_domain": require_pd}
        )
        results = results[:TOPK]

    if not results:
        st.warning("No results. Try disabling Public Domain filter or using a clearer portrait image.")
    else:
        # grid display
        cols = st.columns(min(4, TOPK))
        for i, (rid, meta, score) in enumerate(results):
            c = cols[i % len(cols)]
            with c:
                title  = meta.get("artwork_title_en", "Untitled")
                artist = meta.get("artist_name_en", "Unknown")
                year   = meta.get("year", "?")
                museum = meta.get("museum", "")
                lic    = meta.get("license", "?")
                img_p  = meta.get("image_path")

                st.markdown(f"**{title}**")
                st.caption(f"{artist} • {year} • {museum}")
                st.caption(f"License: {lic} | Score: {score:.3f}")
                if img_p and os.path.exists(img_p):
                    st.image(img_p, use_column_width=True)
                else:
                    st.info("Image not cached locally.")

# Footer
st.markdown("---")
st.caption("MSc Final Project — Embodied Aesthetic Reconstruction | Match-Only Mode (no fusion).")
