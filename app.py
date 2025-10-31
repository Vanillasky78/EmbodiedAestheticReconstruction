# app.py
# ------------------------------------------------------------
# Embodied Aesthetic Reconstruction — Streamlit front-end
# - Live camera / upload
# - CLIP + (optional) Pose/Color re-ranking
# - FAISS index search
# - Safe path handling & robust query image selection
# ------------------------------------------------------------

from __future__ import annotations
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

import streamlit as st

# --- Project paths (robust, absolute) ---
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
IMAGES_DIR   = DATA_DIR / "images"
INTERIM_DIR  = DATA_DIR / "interim"
INDEX_DIR    = BASE_DIR / "indexing"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
LOCKED_FRAME = INTERIM_DIR / "locked_frame.jpg"

# --- Device detect ---
def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = pick_device()

# --- Load CLIP (open_clip) ---
@st.cache_resource(show_spinner=True)
def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = DEVICE):
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, cache_dir=str(BASE_DIR / ".clip_cache")
    )
    model = model.to(device)
    return model, preprocess

# --- Load FAISS index + metadata ---
@st.cache_resource(show_spinner=True)
def load_index(index_dir: Path = INDEX_DIR, images_dir: Path = IMAGES_DIR):
    import faiss
    # faiss
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    ids = np.load(str(index_dir / "ids.npy"))

    # meta list[dict]
    meta_path = index_dir / "meta.json"
    if meta_path.suffix == ".jsonl":
        meta: List[Dict[str, Any]] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    meta.append(json.loads(line))
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # 统一将 meta["image_path"] 规范为可用的绝对路径
    for m in meta:
        p = Path(str(m.get("image_path", "")).strip())
        if p.exists():
            m["image_path"] = str(p.resolve())
            continue
        # 仅文件名
        if p.name and (images_dir / p.name).exists():
            m["image_path"] = str((images_dir / p.name).resolve())
            continue
        # 形如 data/images/xxx.jpg
        try_path = BASE_DIR / p
        if try_path.exists():
            m["image_path"] = str(try_path.resolve())

    return faiss_index, ids, meta

# --- Matcher wrapper ---
@st.cache_resource(show_spinner=False)
def build_matcher():
    from processors.retrieval import Matcher  # uses CLIP image encoder internally
    model, preprocess = load_clip()
    index, ids, meta = load_index()
    return Matcher(index=index, ids=ids, meta=meta, model=model, preprocess=preprocess, device=DEVICE)

# --- Pose overlay (optional) ---
def try_draw_pose_overlay(pil_img: Image.Image) -> Image.Image:
    try:
        from processors.pose import make_pose_detector, PoseEstimator, draw_skeleton
        det = make_pose_detector()  # 自动选择可用/占位
        kps = PoseEstimator.extract_keypoints_pil(pil_img, det)
        if kps is not None:
            return draw_skeleton(pil_img, kps)
    except Exception:
        pass
    return pil_img

# --- Query image selection ---
def get_query_image(upload_file: Any | None) -> Image.Image | None:
    """
    选择用于匹配的查询图片：
      1) 若上传了文件 -> 优先使用
      2) 否则尝试使用最近拍照保存的 locked_frame.jpg
      3) 都没有则返回 None
    """
    # 上传优先
    if upload_file is not None:
        try:
            data = upload_file.getvalue() if hasattr(upload_file, "getvalue") else upload_file.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass

    # 最近拍照
    if LOCKED_FRAME.exists() and LOCKED_FRAME.stat().st_size > 0:
        try:
            return Image.open(LOCKED_FRAME).convert("RGB")
        except Exception:
            pass

    return None

# --- UI helpers ---
def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.markdown("### ⚙️ Matching Weights")
    clip_w  = st.sidebar.slider("CLIP weight", 0.0, 1.0, 1.0, 0.01)
    pose_w  = st.sidebar.slider("Pose weight", 0.0, 1.0, 0.30, 0.01)
    color_w = st.sidebar.slider("Color weight", 0.0, 1.0, 0.20, 0.01)

    st.sidebar.markdown("### 🧰 Filters")
    require_pd = st.sidebar.checkbox("Require Public-Domain license", value=False)

    st.sidebar.markdown("### 📊 Results")
    top_k = st.sidebar.slider("Top-K artworks", 1, 24, 6, 1)

    st.sidebar.markdown("### 💻 Active device: {}".format(DEVICE))

    st.sidebar.markdown("### 🧍 Overlay")
    draw_pose = st.sidebar.checkbox("Draw skeleton on preview (if pose model available)", value=False)

    return dict(clip_w=clip_w, pose_w=pose_w, color_w=color_w,
                require_pd=require_pd, top_k=top_k, draw_pose=draw_pose)

def render_match_cards(items: List[tuple[str, Dict[str, Any], float]]):
    # items: list[(rid, meta, score)]
    if not items:
        st.info("No results. Try lowering filters or Top-K, and check the index files exist.")
        return
    cols = st.columns(min(3, len(items)))
    for i, (rid, m, score) in enumerate(items):
        with cols[i % len(cols)]:
            st.markdown(f"**{m.get('artist_name_en','')} — {m.get('artwork_title_en','')}**")
            p = m.get("image_path", "")
            if p and Path(p).exists():
                st.image(str(p), use_column_width=True)
            st.caption(f"score: {score:.3f} | id: {rid}")

# --- Streamlit App ---
st.set_page_config(page_title="Embodied Aesthetic Reconstruction", layout="wide", page_icon="🎭")

st.title("Embodied Aesthetic Reconstruction")
st.caption("A live interactive system that links your embodied presence to art history.")

controls = sidebar_controls()
matcher  = build_matcher()  # lazy & cached

left, right = st.columns([7, 5], gap="large")

with left:
    st.subheader("Live camera")

    cam = st.camera_input("Take Photo", help="Click 'Take Photo' to capture a still frame for matching.")
    # 保存最近拍照
    if cam is not None:
        try:
            Image.open(io.BytesIO(cam.getvalue())).convert("RGB").save(LOCKED_FRAME)
            # 给用户一个轻量提示
            st.success(f"Saved: {LOCKED_FRAME.relative_to(BASE_DIR)}")
        except Exception as e:
            st.warning(f"Failed saving camera frame: {e}")

    st.markdown("---")
    st.subheader("Or upload an image")
    uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # 预览（可选骨架覆盖）
    q_preview = get_query_image(uploaded)
    if q_preview is not None:
        show = try_draw_pose_overlay(q_preview) if controls["draw_pose"] else q_preview
        st.image(show, caption="Query preview", use_column_width=True)
    else:
        st.info("Please Take Photo or upload an image to enable matching.")

with right:
    st.subheader("Run matching")
    run = st.button("🔍 Match now", type="primary", use_container_width=True)

    st.markdown("### 🎨 Top matches")
    placeholder = st.empty()

    if run:
        # 1) 取查询图
        q_img = get_query_image(uploaded)
        if q_img is None:
            st.error("No query image: please Take Photo or upload an image first.")
        else:
            # 2) 调用检索
            weights = {"w_clip": controls["clip_w"],
                       "w_pose": controls["pose_w"],
                       "w_color": controls["color_w"]}
            filters = {"require_public_domain": controls["require_pd"]}

            start = time.time()
            try:
                items = matcher.search(
                    q_img,
                    k=200,
                    weights=weights,
                    filters=filters,
                    topn=controls["top_k"],
                )
                cost = (time.time() - start) * 1000
                st.caption(f"Search time: {cost:.1f} ms")
                with placeholder.container():
                    render_match_cards(items)
            except Exception as e:
                st.error(f"Search failed: {e}")

    st.markdown("---")
    with st.expander("Tips:"):
        st.markdown(
            """
            - Keep **CLIP** weight at `1.00` while validating similarity; then adjust **Pose/Color** for re-ranking.<br>
            - If results look sparse, **lower** filters or **increase** `Top-K`.<br>
            - Ensure index files exist in `indexing/` and your dataset images are in `data/images/`.
            """,
            unsafe_allow_html=True,
        )
