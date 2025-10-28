"""
app.py
----------------------------------------
Streamlit front-end for the Embodied Aesthetic Reconstruction system.

Flow:
1. Participant stands in front of the camera.
2. The system waits until the participant holds still (~3 seconds).
3. The system captures that exact frame.
4. A pose skeleton is generated from the body keypoints (YOLOv8-Pose).
5. The skeleton image is encoded with OpenCLIP to produce a pose embedding.
6. The pose embedding is compared to a curated portrait dataset of historical artworks.
7. The Top-3 closest portrait matches are displayed with contextual language.

This file is meant to be run with:
    streamlit run app.py

Author: Xinyi Zhang
"""

import time
import os
import streamlit as st
from PIL import Image
from match_pose_to_artworks import run_full_capture_and_match


# ----------------------------------------
# 1. Streamlit page config
# ----------------------------------------

st.set_page_config(
    page_title="Embodied Aesthetic Reconstruction",
    page_icon="🎨",
    layout="wide",
)

st.title("🎭 Embodied Aesthetic Reconstruction")
st.markdown(
    """
    **An AI-driven portrait experience that links your embodied presence to art history.**

    Stand in front of the camera.  
    Hold your pose.  
    The system will watch for stillness (about 3 seconds).  
    When your presence feels stable, it will capture you — not to correct you, not to fix you —  
    but to place you in a lineage of bodies that have already been seen, recorded, and given space.
    """
)

st.markdown("---")


# ----------------------------------------
# 2. Action button
# ----------------------------------------

start = st.button("Start Pose Capture")

if not start:
    st.info("Click **Start Pose Capture** to begin.")
    st.stop()


# ----------------------------------------
# 3. Run full backend pipeline
# ----------------------------------------

with st.spinner("🧍 Detecting pose... Please hold still for ~3 seconds..."):
    start_time = time.time()
    result = run_full_capture_and_match()
    duration = time.time() - start_time

st.success(f"✅ Capture complete in {duration:.1f} seconds.")
st.markdown("---")


# ----------------------------------------
# 4. Show captured participant data
# ----------------------------------------

st.markdown("## 📸 Your Captured Pose")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original Frame")
    if os.path.exists(result["locked_frame_path"]):
        st.image(
            result["locked_frame_path"],
            caption="Captured frame from camera",
            use_container_width=True,
        )
    else:
        st.warning("Captured frame not found on disk.")

with col_right:
    st.subheader("Extracted Pose Skeleton")
    if os.path.exists(result["skeleton_path"]):
        st.image(
            result["skeleton_path"],
            caption="Body keypoints rendered as a skeleton silhouette",
            use_container_width=True,
        )
    else:
        st.warning("Skeleton pose image not found on disk.")

st.markdown("---")


# ----------------------------------------
# 5. Show Top-3 artwork matches
# ----------------------------------------

st.markdown("## 🖼 Top 3 Portrait Matches")

matches = result["results"]

for match in matches:
    artist = match["artist"]
    title = match["title"]
    year = match["year"]
    score = match["score"]
    notes = match["notes_pose"]
    file_name = match["file_name"]

    st.markdown(
        f"""
        ### {match['rank']}. {artist} — *{title}* ({year})
        **Similarity score:** {score:.4f}  
        **Pose reading / embodied attitude:**  
        {notes}
        """
    )

    # Try to display the reference artwork image
    artwork_img_path = os.path.join("data", "images", file_name)

    if os.path.exists(artwork_img_path):
        st.image(
            artwork_img_path,
            caption=f"{title} / {artist}",
            use_container_width=True,
        )
    else:
        st.info(f"(Artwork image not found: {artwork_img_path})")

    st.markdown("---")


# ----------------------------------------
# 6. Final message
# ----------------------------------------

st.markdown("### ✨ Embodied Aesthetic Reconstruction complete")
st.balloons()
st.caption(
    "This system does not grade your body. It does not correct your body. "
    "It acknowledges your body as already belonging to a visual lineage."
)
