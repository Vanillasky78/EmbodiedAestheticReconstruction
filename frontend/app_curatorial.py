# frontend/app_curatorial.py
from __future__ import annotations
import requests
import streamlit as st
from pathlib import Path
import sys

# Try local matcher first
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

try:
    from config import Settings
    from model.pose_matcher import PoseMatcher
    LOCAL = True
except Exception:
    LOCAL = False

st.set_page_config(page_title="EAR — Curatorial", layout="wide")
st.markdown(
    """
    <style>
      .stApp [data-testid=stSidebar] {display: none;}
      .block-container {padding-top: 1rem; padding-bottom: 0.5rem; max-width: 1400px;}
      .stButton>button {font-size:1.1rem; padding:0.5rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>Embodied Aesthetic Reconstruction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:.75;'>Stand still, capture, and see your matched artwork</p>", unsafe_allow_html=True)

# Controls (top, small)
mode = st.session_state.get("mode", "Local" if LOCAL else "Remote")
cols_top = st.columns([1,1,1,1,2,2])
with cols_top[0]:
    mode = st.selectbox("Mode", ["Local", "Remote"], index=0 if LOCAL else 1)
with cols_top[1]:
    topk = st.number_input("Top-K", 1, 10, 3)
with cols_top[2]:
    museum = st.text_input("Museum (optional)", "")
with cols_top[3]:
    btn_clear = st.button("Clear")
with cols_top[4]:
    api_url = st.text_input("API URL (Remote)", "http://127.0.0.1:8000/match")
with cols_top[5]:
    st.caption("Tip: Use Local mode for offline exhibition.")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Camera")
    img_file = st.camera_input("Capture")
    trigger = st.button("Analyze", use_container_width=True)
    if btn_clear:
        st.experimental_rerun()

with right:
    st.subheader("Matched Artworks")
    placeholder = st.empty()

    if trigger and img_file is not None:
        img_bytes = img_file.getvalue()
        if mode == "Local":
            if not LOCAL:
                st.error("Local PoseMatcher not available. Switch to Remote.")
            else:
                matcher = PoseMatcher(settings=Settings())
                results = matcher.match_image_bytes(img_bytes, museum=museum or None, topk=int(topk))
                placeholder.json({"results": results})
        else:
            files = {"file": ("capture.jpg", img_bytes, "image/jpeg")}
            data = {"museum": museum, "topk": str(int(topk))}
            try:
                resp = requests.post(api_url, files=files, data=data, timeout=30)
                resp.raise_for_status()
                placeholder.json(resp.json())
            except Exception as e:
                st.error(f"API error: {e}")
    elif trigger and img_file is None:
        st.warning("Please capture a photo first.")
