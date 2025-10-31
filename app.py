# app.py
# ------------------------------------------------------------
# Embodied Aesthetic Reconstruction (full interactive version)
# - Live camera + pose skeleton overlay (YOLOv8-Pose)
# - Auto-capture on stillness
# - Match captured frame to artworks using CLIP embeddings
# - Adjustable weights: CLIP / Pose / Color
# - Optional Public Domain filter
# ------------------------------------------------------------

import os
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from processors.retrieval import Matcher
from processors.pose import make_pose_detector

# ------------------------------------------------------------
# INITIAL SETUP
# ------------------------------------------------------------

st.set_page_config(
    page_title="Embodied Aesthetic Reconstruction",
    layout="wide",
)

st.title("🪞 Embodied Aesthetic Reconstruction")
st.caption("A live interactive system that links your embodied presence to art history.")

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Active device:** `{device}`")

# ------------------------------------------------------------
# LOAD MATCHER (FAISS index + CLIP model)
# ------------------------------------------------------------

INDEX_DIR = Path("indexing")
DATA_DIR = Path("data/images")

matcher = Matcher(
    faiss_index_path=str(INDEX_DIR / "faiss.index"),
    ids_path=str(INDEX_DIR / "ids.npy"),
    meta_path=str(INDEX_DIR / "meta.json"),
    images_dir=str(DATA_DIR),
    device=device,
)

# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------

with st.sidebar:
    st.subheader("🎛️ Matching Weights")
    w_clip = st.slider("CLIP weight", 0.0, 1.0, 1.0, 0.05)
    w_pose = st.slider("Pose weight", 0.0, 1.0, 0.30, 0.05)
    w_color = st.slider("Color weight", 0.0, 1.0, 0.20, 0.05)

    st.subheader("🧾 Filters")
    require_pd = st.checkbox("Require Public-Domain license", value=False)

    st.subheader("🔢 Results")
    topk = st.slider("Top-K artworks", 3, 20, 6, 1)

    st.markdown("---")
    st.info(
        "Hold still for ~5 seconds to auto-capture.\n"
        "Weights control how CLIP / pose / color influence the final match."
    )

# ------------------------------------------------------------
# VIDEO TRANSFORMER (YOLOv8-Pose)
# ------------------------------------------------------------

pose_model = make_pose_detector()

class PoseOverlay(VideoTransformerBase):
    def __init__(self):
        self.prev_keypoints = None
        self.still_frames = 0
        self.locked = False
        self.last_movement = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = pose_model(img, verbose=False)
        annotated = results[0].plot()

        # Calculate movement
        if len(results[0].keypoints) > 0:
            kpts = results[0].keypoints.cpu().numpy()
            if self.prev_keypoints is not None:
                diff = np.linalg.norm(kpts - self.prev_keypoints)
            else:
                diff = 9999
            self.prev_keypoints = kpts

            if diff < 5:
                self.still_frames += 1
            else:
                self.still_frames = 0

            if self.still_frames > 150 and not self.locked:
                # 5s of stillness at ~30fps
                self.locked = True
                cv2.imwrite("data/interim/locked_frame.jpg", annotated)
                print("Frame captured -> data/interim/locked_frame.jpg")

        return annotated

# ------------------------------------------------------------
# WEBRTC STREAM
# ------------------------------------------------------------

webrtc_streamer(
    key="mirror",
    video_transformer_factory=PoseOverlay,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
)

# ------------------------------------------------------------
# MATCHING SECTION
# ------------------------------------------------------------

st.markdown("---")
st.header("🎨 Top Matches")

saved_path = "data/interim/locked_frame.jpg"
if os.path.exists(saved_path):
    st.success(f"Saved: {saved_path}")
    try:
        results = matcher.search_file(
            img_path=saved_path,
            k=200,
            topn=topk,
            weights=dict(w_clip=w_clip, w_pose=w_pose, w_color=w_color),
            filters=dict(require_public_domain=require_pd),
        )

        if results:
            cols = st.columns(min(len(results), 3))
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    if r.get("image_path") and os.path.exists(r["image_path"]):
                        st.image(r["image_path"])
                    elif r.get("file_name") and os.path.exists(os.path.join("data/images", r["file_name"])):
                        st.image(os.path.join("data/images", r["file_name"]))
                    st.markdown(
                        f"**{r.get('title_en', 'Untitled')}**  \n"
                        f"*{r.get('artist_en', 'Unknown')}*  \n"
                        f"Score: **{r['score']:.3f}**"
                    )
        else:
            st.info("No results found — try adjusting weights or filters.")
    except Exception as e:
        st.error(f"Search failed: {e}")
else:
    st.warning("⚠️ No captured frame yet — please hold still in front of camera.")
