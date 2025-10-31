# app.py — Embodied Aesthetic Reconstruction
# ------------------------------------------------------------
# A live interactive system that links your embodied presence to art history.
# ------------------------------------------------------------

import streamlit as st
import torch
import time
from pathlib import Path
from PIL import Image
import numpy as np

# --- internal modules ---
from processors.pose import make_pose_detector, PoseEstimator, draw_skeleton
from processors.retrieval import Matcher
from processors.generate_embeddings import load_clip_model

# --- basic config ---
st.set_page_config(page_title="Embodied Aesthetic Reconstruction", layout="wide")

# ------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------
st.title("🪞 Embodied Aesthetic Reconstruction")
st.caption("A live interactive system that links your embodied presence to art history.")

# ------------------------------------------------------------
# Sidebar — configuration controls
# ------------------------------------------------------------
st.sidebar.subheader("⚙️ Matching Weights")
w_clip = st.sidebar.slider("CLIP weight", 0.0, 1.0, 1.0, 0.05)
w_pose = st.sidebar.slider("Pose weight", 0.0, 1.0, 0.3, 0.05)
w_color = st.sidebar.slider("Color weight", 0.0, 1.0, 0.2, 0.05)

st.sidebar.subheader("🖼️ Filters")
require_pd = st.sidebar.checkbox("Require Public-Domain license", False)
topk = st.sidebar.slider("Top-K artworks", 1, 12, 6)

# ------------------------------------------------------------
# Device setup
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

st.sidebar.text(f"Active device: {DEVICE}")

# ------------------------------------------------------------
# Load CLIP + FAISS index
# ------------------------------------------------------------
@st.cache_resource
def load_system():
    import faiss
    import json
    from processors.generate_embeddings import load_clip_model

    # load model + preprocessor
    model, preprocess = load_clip_model(device=DEVICE)

    # load index and meta
    index_path = Path("indexing/faiss.index")
    ids_path = Path("indexing/ids.npy")
    meta_path = Path("indexing/meta.json")

    index = faiss.read_index(str(index_path))
    ids = np.load(ids_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    matcher = Matcher(index=index, ids=ids, meta=meta, model=model, preprocess=preprocess, device=DEVICE)
    return matcher, model, preprocess

matcher, model, preprocess = load_system()

# ------------------------------------------------------------
# Load pose model (YOLOv8-Pose)
# ------------------------------------------------------------
pose_model = make_pose_detector("yolov8n-pose.pt", device=DEVICE)

# ------------------------------------------------------------
# Live mirror setup
# ------------------------------------------------------------
st.subheader("🎥 Live Mirror + Pose (Auto-capture on stillness)")

pose_conf = st.slider("Pose confidence threshold", 0.0, 1.0, 0.4, 0.05)
mirror_preview = st.checkbox("Mirror preview", True)
stillness_time = st.number_input("Stillness required (seconds)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
motion_sensitivity = st.slider("Motion sensitivity (lower = more sensitive)", 0.5, 5.0, 2.0, 0.1)

manual_save = st.button("📸 Save frame manually")

# --- initialize camera ---
camera = st.camera_input("Hold still for ~5 seconds to auto-capture", key="camera")

# Persistent state
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_change" not in st.session_state:
    st.session_state.last_change = time.time()

# ------------------------------------------------------------
# Utility: detect stillness
# ------------------------------------------------------------
def is_still(curr, prev, threshold=motion_sensitivity):
    if curr is None or prev is None:
        return False
    diff = np.mean(np.abs(np.array(curr).astype(float) - np.array(prev).astype(float)))
    return diff < threshold * 5  # adjustable tolerance

# ------------------------------------------------------------
# Handle input frame
# ------------------------------------------------------------
if camera is not None:
    frame = Image.open(camera)
    frame_rgb = frame.convert("RGB")

    # Draw skeleton overlay (mirror view)
    keypoints = PoseEstimator.extract_keypoints_pil(frame_rgb, pose_model)
    if keypoints is not None:
        overlay = draw_skeleton(frame_rgb, keypoints)
        if mirror_preview:
            overlay = overlay.transpose(Image.FLIP_LEFT_RIGHT)
        st.image(overlay, caption="Live pose mirror", use_container_width=True)
    else:
        st.image(frame_rgb, caption="Live feed", use_container_width=True)

    # Auto-capture logic
    last_frame = st.session_state.last_frame
    still = is_still(frame_rgb, last_frame, threshold=motion_sensitivity)
    now = time.time()

    if still:
        delta = now - st.session_state.last_change
        st.info(f"STILL… | Δ={delta:.2f}s | threshold≈{stillness_time:.2f}")
        if delta > stillness_time or manual_save:
            out_path = Path("data/interim/locked_frame.jpg")
            frame_rgb.save(out_path)
            st.success(f"✅ Captured frame saved to {out_path}")
            # Perform matching
            try:
                st.subheader("🎨 Top matches")
                q_img = Image.open(out_path).convert("RGB")
                results = matcher.search(
                    q_img,
                    topn=topk,
                    weights={"w_clip": w_clip, "w_pose": w_pose, "w_color": w_color},
                    filters={"require_public_domain": require_pd},
                )
                if not results:
                    st.warning("No matches found.")
                else:
                    cols = st.columns(min(3, len(results)))
                    for i, (rid, meta, score) in enumerate(results):
                        with cols[i % len(cols)]:
                            if "image_path" in meta and Path(meta["image_path"]).exists():
                                img = Image.open(meta["image_path"])
                                st.image(img, caption=f"{meta.get('title_en', 'Untitled')} — {meta.get('artist_en', '')} ({score:.3f})")
                            else:
                                st.markdown(f"**{meta.get('title_en', 'Untitled')}**<br>{meta.get('artist_en', '')} ({score:.3f})", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Search failed: {e}")
            st.session_state.last_change = now  # reset timer
    else:
        st.info("MOVE… (motion detected)")
        st.session_state.last_change = now

    # Update reference
    st.session_state.last_frame = frame_rgb

else:
    st.warning("📷 Please enable your camera to begin live matching.")
