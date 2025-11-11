# frontend/app_frontend.py
from __future__ import annotations

import requests
import streamlit as st
from PIL import Image

from pathlib import Path
import io

# Optional local import (works without running FastAPI)
import sys
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

try:
    from config import Settings
    from model.pose_matcher import PoseMatcher
    LOCAL_AVAILABLE = True
except Exception:
    LOCAL_AVAILABLE = False

st.set_page_config(page_title="EAR — Camera Matching", layout="wide")
st.title("Embodied Aesthetic Reconstruction — Frontend")

with st.sidebar:
    st.markdown("### Mode")
    mode_options = ["Local (direct import)", "Remote API"]
    default_idx = 0 if LOCAL_AVAILABLE else 1
    mode = st.radio("How to run matching?", mode_options, index=default_idx)
    museum = st.text_input("Museum (optional, e.g., local / met)", value="")
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=3, step=1)
    api_url = st.text_input("API URL (for Remote API)", value="http://127.0.0.1:8000/match")
    st.caption("Tip: if Local import fails, switch to Remote API.")

st.markdown("#### Camera")
img_file = st.camera_input("Take a photo, then click Analyze")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Your Photo")
    if img_file is not None:
        st.image(img_file)

analyze = st.button("Analyze")

with col2:
    st.markdown("#### Results")
    if analyze and img_file is not None:
        img_bytes = img_file.getvalue()

        if mode.startswith("Local"):
            if not LOCAL_AVAILABLE:
                st.error("Local PoseMatcher not available. Switch to Remote API.")
            else:
                st.info("Running in Local mode (PoseMatcher inside Streamlit).")
                settings = Settings()
                matcher = PoseMatcher(settings=settings)
                results = matcher.match_image_bytes(img_bytes, museum=museum or None, topk=int(topk))
                st.json({"results": results})
                if results:
                    r0 = results[0]
                    title = r0.get("title", r0.get("filename", ""))
                    artist = r0.get("artist", "")
                    sim = r0.get("similarity", 0.0)
                    st.markdown(f"**Top-1:** {artist} — {title} (sim={sim:.3f})")
        else:
            st.info("Calling Remote API.")
            files = {"file": ("capture.jpg", img_bytes, "image/jpeg")}
            data = {"museum": museum, "topk": str(int(topk))}
            try:
                resp = requests.post(api_url, files=files, data=data, timeout=30)
                resp.raise_for_status()
                st.json(resp.json())
            except Exception as e:
                st.error(f"Request failed: {e}")
    elif analyze and img_file is None:
        st.warning("Please capture a photo first.")
