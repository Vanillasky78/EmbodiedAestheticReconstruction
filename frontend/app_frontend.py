"""
Streamlit Frontend for Embodied Aesthetic Reconstruction

Two modes:
1) Local (direct import)  – uses PoseMatcher inside this process
2) Remote API             – calls FastAPI backend at /match

The layout matches your previous design: sidebar for controls,
camera in the center, results on the right.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# Make backend importable when running "streamlit run frontend/app_frontend.py"
# ---------------------------------------------------------------------
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.config import Settings  # noqa: E402
from backend.model.pose_matcher import PoseMatcher  # noqa: E402


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@st.cache_resource
def get_local_matcher() -> PoseMatcher:
    """Create a single PoseMatcher instance for Local mode."""
    settings = Settings()
    return PoseMatcher(settings)


def call_remote_api(
    api_url: str,
    image: Image.Image,
    museum: str,
    topk: int,
):
    """Send the image to the FastAPI backend."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"image": ("capture.png", buf, "image/png")}
    data = {"museum": museum, "topk": str(topk)}

    resp = requests.post(api_url, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_local_matcher(
    image: Image.Image,
    museum: str,
    topk: int,
):
    matcher = get_local_matcher()
    raw = matcher.match_pil(image, museum=museum or None, topk=topk)
    # Shape into the same JSON-like structure as the backend
    return {
        "museum": museum or matcher.default_museum,
        "topk": topk,
        "results": raw,
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="EAR — Camera Matching",
    page_icon="🌀",
    layout="wide",
)

st.title("Embodied Aesthetic Reconstruction — Frontend")

# ------------- Sidebar controls -------------
with st.sidebar:
    st.header("Mode")

    mode = st.radio(
        "How to run matching?",
        options=["Local (direct import)", "Remote API"],
        index=1,
    )

    museum = st.text_input(
        "Museum (optional, e.g., local / met)",
        value="local",
    )

    topk = st.number_input(
        "Top-K",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    api_url = st.text_input(
        "API URL (for Remote API)",
        value="http://127.0.0.1:8000/match",
    )

    st.caption("Tip: if Local import fails, switch to Remote API.")

# ------------- Camera + Results -------------
col_cam, col_res = st.columns([2, 2])

with col_cam:
    st.subheader("Camera")
    st.write("Take a photo, then click Analyze")

    camera_img = st.camera_input("Camera", label_visibility="collapsed")

    if camera_img is not None:
        st.image(camera_img, caption="Your Photo", use_column_width=True)

with col_res:
    st.subheader("Results")

    if camera_img is None:
        st.info("Take a photo first, then click **Analyze**.")
    else:
        if st.button("Analyze", type="primary"):
            try:
                pil_img = Image.open(camera_img).convert("RGB")

                if mode == "Remote API":
                    st.info("Calling Remote API…")
                    resp_json = call_remote_api(
                        api_url=api_url,
                        image=pil_img,
                        museum=museum,
                        topk=int(topk),
                    )
                else:
                    st.info("Running Local matcher…")
                    resp_json = run_local_matcher(
                        image=pil_img,
                        museum=museum,
                        topk=int(topk),
                    )

                # Pretty JSON on the right
                st.json(resp_json)

                # Optionally show the top-K thumbnails below (if files exist locally)
                results = resp_json.get("results", [])
                if results:
                    st.markdown("### Matched Artworks")
                    img_cols = st.columns(len(results))
                    for c, r in zip(img_cols, results):
                        filename = r.get("filename")
                        title = r.get("title") or "(untitled)"
                        artist = r.get("artist") or "(unknown artist)"
                        score = float(r.get("score", 0.0))

                        with c:
                            img_path = PROJECT_ROOT / "data" / "local" / "images" / filename
                            if img_path.exists():
                                st.image(str(img_path), use_column_width=True)
                            st.caption(f"**{title}**  \n{artist}  \nscore: {score:.3f}")

            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
                st.code(str(exc))
