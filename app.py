# app.py
# ============================================================
# Embodied Aesthetic Reconstruction – Streamlit App (clean)
# - Camera or upload a photo
# - Build a Matcher and retrieve top-K artworks
# - Uses indexing/{faiss.index, ids.npy, meta.json} built by build_index.py
# - Pose/Color re-ranking are optional and degrade gracefully
# ============================================================

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# --- external libs we rely on ---
import torch
import faiss
import open_clip

# our modules
from processors.retrieval import Matcher  # uses Pose/Color if available
from processors.pose import make_pose_detector, PoseEstimator, draw_skeleton  # safe fallbacks


# ------------------------------- Paths & constants -------------------------------

APP_TITLE = "Embodied Aesthetic Reconstruction"
LOCKED_FRAME = Path("data/interim/locked_frame.jpg")
INDEX_DIR = Path("indexing")
FAISS_PATH = INDEX_DIR / "faiss.index"
IDS_PATH = INDEX_DIR / "ids.npy"
META_PATH = INDEX_DIR / "meta.json"

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"


# ------------------------------- Utilities -------------------------------

@st.cache_resource(show_spinner=False)
def get_device() -> str:
    """Pick a sensible default device."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner=True)
def load_clip(model_name: str = DEFAULT_MODEL,
              pretrained: str = DEFAULT_PRETRAINED,
              device: str = "cpu"):
    """Load OpenCLIP model + preprocess."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    return model, preprocess


@st.cache_resource(show_spinner=True)
def load_index() -> Tuple[Any, np.ndarray, List[Dict[str, Any]]]:
    """Load FAISS + ids + meta.json (built by indexing/build_index.py)."""
    if not (FAISS_PATH.exists() and IDS_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError(
            "Index not found. Please build it first:\n"
            "python indexing/build_index.py --dataset_jsonl data/interim/portrait_art_dataset.jsonl "
            "--images_dir data/images --out_dir indexing"
        )
    index = faiss.read_index(str(FAISS_PATH))
    ids = np.load(str(IDS_PATH))
    meta = json.loads(Path(META_PATH).read_text(encoding="utf-8"))
    return index, ids, meta


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def pil_from_upload(upload) -> Image.Image | None:
    if upload is None:
        return None
    return Image.open(io.BytesIO(upload.getvalue())).convert("RGB")


def save_pil(img: Image.Image, path: Path):
    ensure_dir(path)
    img.save(str(path))


def render_matches(matches: List[Tuple[str, Dict[str, Any], float]], header="🎨 Top matches"):
    st.subheader(header)
    if not matches:
        st.info("No results. Try adjusting weights/filters or take another photo.")
        return
    cols = st.columns(min(3, len(matches)))
    for i, (rid, meta, score) in enumerate(matches):
        c = cols[i % len(cols)]
        img_path = meta.get("image_path") or meta.get("file_path") or ""
        title = meta.get("title_en") or meta.get("title") or ""
        artist = meta.get("artist_en") or meta.get("artist") or ""
        caption = f"{title} — {artist}\nScore: {score:.3f}"
        if img_path and Path(img_path).exists():
            c.image(img_path, use_container_width=True, caption=caption)
        else:
            c.write(caption)


def maybe_draw_pose(pil_img: Image.Image, draw_pose: bool) -> Image.Image:
    if not draw_pose:
        return pil_img
    try:
        detector = make_pose_detector()  # safe fallback to Dummy
        kps = PoseEstimator.extract_keypoints_pil(pil_img, detector)
        if kps is not None:
            return draw_skeleton(pil_img, kps, radius=4)
    except Exception:
        pass
    return pil_img


# ------------------------------- UI -------------------------------

def sidebar_controls():
    st.sidebar.markdown("### ⚙️ Matching Weights")
    w_clip = st.sidebar.slider("CLIP weight", 0.0, 1.0, 1.0, 0.01)
    w_pose = st.sidebar.slider("Pose weight", 0.0, 1.0, 0.30, 0.01)
    w_color = st.sidebar.slider("Color weight", 0.0, 1.0, 0.20, 0.01)

    st.sidebar.markdown("### 🧰 Filters")
    require_pd = st.sidebar.checkbox("Require Public-Domain license", value=False)

    st.sidebar.markdown("### 📊 Results")
    topk = st.sidebar.slider("Top-K artworks", 1, 24, 6, 1)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Active device: **{st.session_state['device']}**")

    st.sidebar.markdown("### 🧍 Overlay")
    draw_pose = st.sidebar.checkbox("Draw skeleton on preview (if pose model available)", value=False)

    return dict(
        w_clip=w_clip, w_pose=w_pose, w_color=w_color,
        require_pd=require_pd, topk=topk, draw_pose=draw_pose
    )


# ------------------------------- App main -------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A live interactive system that links your embodied presence to art history.")

    # cache device
    if "device" not in st.session_state:
        st.session_state["device"] = get_device()
    device = st.session_state["device"]

    # Load model + index
    with st.spinner("Loading model & index..."):
        model, preprocess = load_clip(device=device)
        index, ids, meta = load_index()
    matcher = Matcher(index=index, ids=ids, meta=meta, model=model, preprocess=preprocess, device=device)

    # Sidebar controls
    cfg = sidebar_controls()

    # Camera / Upload area
    left, right = st.columns([3, 2], vertical_alignment="top")

    with left:
        st.markdown("#### Live camera")
        snapshot = st.camera_input("Take Photo", key="cam", help="Grant camera permission, then click 'Take Photo'.")
        st.markdown("#### Or upload an image")
        upload = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        q_img: Image.Image | None = None
        source = None
        if snapshot is not None:
            q_img = pil_from_upload(snapshot)
            source = "camera"
        elif upload is not None:
            q_img = pil_from_upload(upload)
            source = "upload"

        if q_img is not None:
            prev = maybe_draw_pose(q_img, cfg["draw_pose"])
            st.image(prev, use_container_width=True, caption=f"Preview ({source})")
            # save locked frame
            save_pil(q_img, LOCKED_FRAME)
            st.success(f"Saved: {LOCKED_FRAME}")

    with right:
        st.markdown("#### Run matching")
        go = st.button("🔍 Match now", type="primary", use_container_width=True)
        if go:
            # decide the query image
            q: Image.Image | None = None
            if LOCKED_FRAME.exists():
                q = Image.open(str(LOCKED_FRAME)).convert("RGB")
            elif upload is not None:
                q = pil_from_upload(upload)
            elif snapshot is not None:
                q = pil_from_upload(snapshot)

            if q is None:
                st.error("No image to search. Take a photo or upload one.")
            else:
                weights = {"w_clip": cfg["w_clip"], "w_pose": cfg["w_pose"], "w_color": cfg["w_color"]}
                filters = {"require_public_domain": cfg["require_pd"]}
                try:
                    matches = matcher.search(q, k=200, weights=weights, filters=filters, topn=int(cfg["topk"]))
                    render_matches(matches)
                except Exception as e:
                    st.error(f"Search failed: {e}")

        st.markdown("---")
        st.info("Tips:\n- Increase Pose/Color weight after you confirm CLIP matches look reasonable.\n"
                "- If no results, try lowering filters or Top-K, and check the index files exist.")

    # Footer
    st.markdown("---")
    st.caption("© 2025 EAR — OpenCLIP + FAISS | Pose optional (YOLOv8-Pose if available)")


if __name__ == "__main__":
    main()
