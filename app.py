# app.py
# Embodied Aesthetic Reconstruction — Streamlit App (M4 / macOS ready)
# -------------------------------------------------------------------
# Features
# - Robust path resolution (always finds data/images next to this file)
# - Skips iCloud placeholder files (*.icloud)
# - Upload or camera capture; optional skeleton overlay
# - CLIP / Color / Pose tri-fusion with user-adjustable weights
# - Public-Domain filter (if present in metadata)
# - Auto-caching of models and dataset embeddings
# - Graceful fallbacks when optional deps/models are missing
# -------------------------------------------------------------------

from __future__ import annotations
import os
import io
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# ---------- Device / Torch ----------
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

def get_device() -> str:
    if not TORCH_OK:
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()

# ---------- Paths ----------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
INTERIM_DIR = DATA_DIR / "interim"
CACHE_DIR = APP_DIR / ".clip_cache"
CACHE_DIR.mkdir(exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV = DATA_DIR / "portrait_works.csv"

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="Embodied Aesthetic Reconstruction",
    page_icon="🖼️",
    layout="wide",
)

# ---------- Utils ----------
def is_icloud_placeholder(p: Path) -> bool:
    # iCloud keeps placeholders like `xxx.jpg.icloud` (sometimes hidden)
    return p.suffix == ".icloud" or p.name.endswith(".icloud")

def load_image_safe(path: Path) -> Optional[Image.Image]:
    if not path.exists() or is_icloud_placeholder(path):
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def center_crop_long_edge(im: Image.Image, size: int = 512) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    return im.resize((size, size), Image.BICUBIC)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Matching Weights")
    w_clip = st.slider("CLIP weight", 0.0, 1.0, 1.00, 0.05)
    w_pose = st.slider("Pose weight", 0.0, 1.0, 0.30, 0.05)
    w_color = st.slider("Color weight", 0.0, 1.0, 0.20, 0.05)

    st.markdown("---")
    st.header("🔎 Filters")
    require_public = st.checkbox("Require Public-Domain license", value=False)

    st.markdown("---")
    st.header("📊 Results")
    top_k = st.slider("Top-K artworks", 1, 20, 6, 1)

    st.markdown("---")
    st.header("💻 Active device")
    st.success(f"{DEVICE}")

    st.markdown("---")
    st.header("🧍 Overlay")
    overlay_skeleton = st.checkbox("Draw skeleton on preview (if available)", value=True)

# ---------- Load metadata ----------
def load_metadata() -> pd.DataFrame:
    if METADATA_CSV.exists():
        try:
            df = pd.read_csv(METADATA_CSV)
            return df
        except Exception as e:
            st.warning(f"Could not read metadata CSV ({METADATA_CSV.name}): {e}")
    return pd.DataFrame({"filename": [], "title": [], "artist": [], "year": [], "license": []})

META = load_metadata()

# ---------- Collect dataset images ----------
def list_dataset_images() -> List[Path]:
    if not IMAGES_DIR.exists():
        return []
    all_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        all_files += list(IMAGES_DIR.glob(ext))
    files = [p for p in all_files if not is_icloud_placeholder(p)]
    return sorted(files)

DATASET_FILES = list_dataset_images()

# ---------- OpenCLIP ----------
@st.cache_resource(show_spinner=False)
def load_openclip():
    try:
        import open_clip
    except Exception as e:
        st.error(f"open_clip_torch is not installed: {e}")
        return None, None, None
    try:
        model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=DEVICE
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, preprocess, tokenizer
    except Exception as e:
        st.error(f"Failed to load OpenCLIP: {e}")
        return None, None, None

MODEL_CLIP, PRE_CLIP, TOKENIZER = load_openclip()

def tensor_from_pil_clip(im: Image.Image):
    if PRE_CLIP is None:
        return None
    t = PRE_CLIP(im).unsqueeze(0)
    if TORCH_OK:
        t = t.to(DEVICE)
    return t

@st.cache_data(show_spinner=False)
def embed_image_clip(img_bytes: bytes) -> Optional[np.ndarray]:
    if MODEL_CLIP is None:
        return None
    try:
        im = pil_from_bytes(img_bytes)
        im = center_crop_long_edge(im, size=224)
        x = tensor_from_pil_clip(im)
        with torch.no_grad():
            feat = MODEL_CLIP.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32")[0]
    except Exception:
        return None

# ---------- Dataset Embeddings ----------
EMB_PATH = CACHE_DIR / "embeddings.npy"
IDS_PATH = CACHE_DIR / "ids.json"
POSE_PATH = CACHE_DIR / "pose.npy"      # (optional)
COLOR_PATH = CACHE_DIR / "color.npy"    # (optional)

def _color_feature(im: Image.Image) -> np.ndarray:
    # HSV histogram (H 32, S 16, V 8) → L2-normalize → float32
    im = im.resize((256, 256), Image.BILINEAR)
    arr = np.array(im.convert("HSV"))
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    hist_h, _ = np.histogram(h, bins=32, range=(0, 255), density=True)
    hist_s, _ = np.histogram(s, bins=16, range=(0, 255), density=True)
    hist_v, _ = np.histogram(v, bins=8, range=(0, 255), density=True)
    feat = np.concatenate([hist_h, hist_s, hist_v]).astype("float32")
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm

def _pose_feature_from_kpts(kpts_xyv: np.ndarray) -> np.ndarray:
    # kpts: (N, 3) x,y,score → normalize to unit box; flatten; zero if invalid
    if kpts_xyv is None or len(kpts_xyv) == 0:
        return np.zeros(34, dtype="float32")  # 17*2 for COCO
    xy = kpts_xyv[:, :2]
    min_xy = xy.min(0)
    max_xy = xy.max(0)
    wh = np.maximum(max_xy - min_xy, 1e-6)
    xy_n = (xy - min_xy) / wh
    feat = xy_n.reshape(-1).astype("float32")
    if feat.shape[0] < 34:
        feat = np.pad(feat, (0, 34 - feat.shape[0]))
    else:
        feat = feat[:34]
    n = np.linalg.norm(feat) + 1e-8
    return feat / n

def _detect_pose_pil(im: Image.Image) -> Optional[np.ndarray]:
    # YOLOv8n-pose via ultralytics
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        model = YOLO("yolov8n-pose.pt")
        arr = np.array(im.convert("RGB"))
        res = model.predict(source=arr, imgsz=512, conf=0.25, verbose=False,
                            device=0 if DEVICE != "cpu" else "cpu")
        if not res or len(res[0].keypoints) == 0:
            return None
        kpts = res[0].keypoints.xy.cpu().numpy()  # (persons, 17, 2)
        if kpts.ndim == 4:
            kpts = kpts[0]
        if kpts.ndim == 3:
            k = kpts[0]
        else:
            return None
        v = np.ones((k.shape[0], 1), dtype="float32")
        kxyv = np.concatenate([k.astype("float32"), v], axis=1)
        return _pose_feature_from_kpts(kxyv)
    except Exception:
        return None

def _dataset_pose_feature(path: Path) -> Optional[np.ndarray]:
    im = load_image_safe(path)
    if im is None:
        return None
    im_small = center_crop_long_edge(im, 512)
    return _detect_pose_pil(im_small)

@st.cache_data(show_spinner=True, persist=True)
def build_dataset_embeddings(files: List[Path]) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (clip_embeds, ids, pose_feats, color_feats)"""
    ids = [str(p.relative_to(APP_DIR)) for p in files]
    clip_feats, pose_feats, color_feats = [], [], []

    # Try to load cached
    if EMB_PATH.exists() and IDS_PATH.exists():
        try:
            cached_ids = json.loads(IDS_PATH.read_text())
            if cached_ids == ids:
                clip_feats = np.load(EMB_PATH)
                pose_feats = np.load(POSE_PATH) if POSE_PATH.exists() else None
                color_feats = np.load(COLOR_PATH) if COLOR_PATH.exists() else None
                return clip_feats, ids, pose_feats, color_feats
        except Exception:
            pass

    if MODEL_CLIP is None:
        st.error("OpenCLIP not ready; cannot build dataset embeddings.")
        return np.zeros((0, 512), dtype="float32"), ids, None, None

    progress = st.progress(0.0, text="Building dataset embeddings/features…")
    for i, p in enumerate(files):
        progress.progress((i + 1) / max(1, len(files)))
        im = load_image_safe(p)
        if im is None:
            clip_feats.append(np.zeros(512, dtype="float32"))
            pose_feats.append(np.zeros(34, dtype="float32"))
            color_feats.append(np.zeros(56, dtype="float32"))  # 32+16+8
            continue

        # CLIP
        try:
            im_clip = center_crop_long_edge(im, 224)
            x = tensor_from_pil_clip(im_clip)
            with torch.no_grad():
                z = MODEL_CLIP.encode_image(x)
                z = z / z.norm(dim=-1, keepdim=True)
            clip_feats.append(z.cpu().numpy().astype("float32")[0])
        except Exception:
            clip_feats.append(np.zeros(512, dtype="float32"))

        # Color
        try:
            color_feats.append(_color_feature(im))
        except Exception:
            color_feats.append(np.zeros(56, dtype="float32"))

        # Pose (optional)
        pf = _dataset_pose_feature(p)
        if pf is None:
            pf = np.zeros(34, dtype="float32")
        pose_feats.append(pf)

    clip_arr = np.vstack(clip_feats).astype("float32") if clip_feats else np.zeros((0, 512), dtype="float32")
    pose_arr = np.vstack(pose_feats).astype("float32") if pose_feats else None
    color_arr = np.vstack(color_feats).astype("float32") if color_feats else None

    # Cache
    try:
        np.save(EMB_PATH, clip_arr)
        np.save(POSE_PATH, pose_arr)
        np.save(COLOR_PATH, color_arr)
        IDS_PATH.write_text(json.dumps(ids))
    except Exception:
        pass

    return clip_arr, ids, pose_arr, color_arr

# Build dataset features
if len(DATASET_FILES) == 0:
    st.error(f"❌ No images found or folder is empty: {IMAGES_DIR}\nMake sure iCloud files are downloaded locally (no .icloud).")
else:
    st.success(f"✅ Dataset images: {len(DATASET_FILES)}")

CLIP_DS, IDS, POSE_DS, COLOR_DS = build_dataset_embeddings(DATASET_FILES)

# ---------- Similarity ----------
def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a[None, :]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a @ b.T)

def rank_results(
    q_clip: Optional[np.ndarray],
    q_pose: Optional[np.ndarray],
    q_color: Optional[np.ndarray],
    topk: int,
    weights: Tuple[float, float, float],
) -> List[Tuple[int, float]]:
    w_clip, w_pose, w_color = weights
    score = np.zeros(len(IDS), dtype="float32")

    if q_clip is not None and CLIP_DS.shape[0] == len(IDS):
        score += w_clip * cos_sim(q_clip, CLIP_DS)[0]

    if q_pose is not None and POSE_DS is not None:
        score += w_pose * cos_sim(q_pose, POSE_DS)[0]

    if q_color is not None and COLOR_DS is not None:
        score += w_color * cos_sim(q_color, COLOR_DS)[0]

    idx = np.argsort(-score)[:topk]
    return [(int(i), float(score[i])) for i in idx]

# ---------- Pose overlay ----------
def draw_skeleton_overlay(im: Image.Image) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """Returns (image_with_overlay, pose_feature or None)"""
    try:
        from ultralytics import YOLO
        import cv2
    except Exception:
        return im, None

    model = YOLO("yolov8n-pose.pt")
    arr = np.array(im.convert("RGB"))
    res = model.predict(source=arr, imgsz=512, conf=0.25, verbose=False,
                        device=0 if DEVICE != "cpu" else "cpu")
    if not res or len(res[0].keypoints) == 0:
        return im, None

    # Draw on a copy
    canvas = arr.copy()
    K = res[0].keypoints
    pts = K.xy.cpu().numpy()
    if pts.ndim == 4:
        pts = pts[0]
    if pts.ndim == 3:
        pts = pts[0]  # take first person

    PAIRS = [
        (5, 7), (7, 9), (6, 8), (8,10),      # arms
        (11,13), (13,15), (12,14), (14,16),  # legs
        (5,6), (11,12), (5,11), (6,12)       # torso
    ]

    try:
        import cv2
        for (a, b) in PAIRS:
            xa, ya = pts[a]
            xb, yb = pts[b]
            cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 3)
        for (x, y) in pts:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
        im_draw = Image.fromarray(canvas)
    except Exception:
        im_draw = im

    v = np.ones((pts.shape[0], 1), dtype="float32")
    kxyv = np.concatenate([pts.astype("float32"), v], axis=1)
    pf = _pose_feature_from_kpts(kxyv)
    return im_draw, pf

# ---------- UI — Main ----------
st.title("Embodied Aesthetic Reconstruction")
st.caption("Camera / Upload → CLIP + Pose + Color matching → Top-K artworks")

# Status block
status_cols = st.columns(3)
with status_cols[0]:
    if IMAGES_DIR.exists():
        st.success(f"Images dir: {IMAGES_DIR.relative_to(APP_DIR)}")
    else:
        st.error(f"Missing folder: {IMAGES_DIR}")
with status_cols[1]:
    st.info(f"Dataset files: {len(DATASET_FILES)}")
with status_cols[2]:
    st.info(f"Models: OpenCLIP={'✔︎' if MODEL_CLIP else '✖︎'} · Pose(ultralytics)={'✔︎' if POSE_DS is not None else '◦'}")

st.markdown("---")

# Capture / Upload
left, right = st.columns([1, 1])
query_img: Optional[Image.Image] = None
query_pose_feat: Optional[np.ndarray] = None

with left:
    st.subheader("📷 Camera (auto center-crop)")
    cam = st.camera_input("Take a photo (allow permission first)", key="camera")
    if cam is not None:
        try:
            img = pil_from_bytes(cam.getvalue())
            img = center_crop_long_edge(img, 640)
            out_path = INTERIM_DIR / "locked_frame.jpg"
            img.save(out_path, quality=92)
            st.success(f"Saved: {out_path.as_posix()}")
            if overlay_skeleton:
                img_draw, q_pose = draw_skeleton_overlay(img)
                st.image(img_draw, caption="Camera preview", use_container_width=True)
                query_img = img_draw
                query_pose_feat = q_pose
            else:
                st.image(img, caption="Camera preview", use_container_width=True)
                query_img = img
        except Exception as e:
            st.error(f"Camera decode failed: {e}")

with right:
    st.subheader("🖼️ Or upload an image")
    up = st.file_uploader("JPG/PNG", type=["jpg", "jpeg", "png"])
    if up is not None:
        try:
            img = pil_from_bytes(up.getvalue())
            img = center_crop_long_edge(img, 640)
            if overlay_skeleton:
                img_draw, q_pose = draw_skeleton_overlay(img)
                st.image(img_draw, caption="Uploaded preview", use_container_width=True)
                query_img = img_draw
                query_pose_feat = q_pose
            else:
                st.image(img, caption="Uploaded preview", use_container_width=True)
                query_img = img
        except Exception as e:
            st.error(f"Upload decode failed: {e}")

# Matching button
st.markdown("---")
run = st.button("🔎 Run Matching")

def df_filter_public(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "license" not in df.columns:
        return df
    if not require_public:
        return df
    return df[df["license"].astype(str).str.contains("Public Domain", case=False, na=False)]

def display_results(idx_scores: List[Tuple[int, float]], k: int):
    if len(idx_scores) == 0:
        st.warning("No results to display.")
        return
    cols = st.columns(min(3, k))
    for i, (idx, sc) in enumerate(idx_scores[:k]):
        fn_rel = IDS[idx]
        p = (APP_DIR / fn_rel).resolve()
        im = load_image_safe(p)
        # attach metadata if available
        row = None
        if not META.empty and "filename" in META.columns:
            rowset = META[META["filename"].astype(str).str.contains(Path(fn_rel).name)]
            if len(rowset) > 0:
                row = rowset.iloc[0]
        if row is not None:
            title = str(row.get("title", "")).strip()
            artist = str(row.get("artist", "")).strip()
            year = str(row.get("year", "")).strip()
            meta_txt = f"{title} — {artist} ({year})"
        else:
            meta_txt = Path(fn_rel).name

        with cols[i % len(cols)]:
            if im is not None:
                st.image(im, use_container_width=True)
            st.markdown(f"**Score:** {sc:.3f}")
            st.caption(meta_txt)

if run:
    if query_img is None:
        st.warning("Please take a photo or upload an image first.")
    else:
        st.write("Computing embeddings…")
        t0 = time.time()

        # Query CLIP
        q_clip = None
        if MODEL_CLIP is not None:
            try:
                buf = io.BytesIO()
                query_img.save(buf, format="JPEG", quality=92)
                q_clip = embed_image_clip(buf.getvalue())
            except Exception:
                q_clip = None

        # Query color
        try:
            q_color = _color_feature(query_img)
        except Exception:
            q_color = None

        # Query pose (may already exist from overlay step)
        if query_pose_feat is None and w_pose > 0:
            query_pose_feat = _detect_pose_pil(center_crop_long_edge(query_img, 512))

        # If pose backend missing, zero its weight
        w_pose_eff = w_pose if query_pose_feat is not None and POSE_DS is not None else 0.0
        if w_pose > 0 and w_pose_eff == 0.0:
            st.info("No pose feature detected or dataset pose not precomputed; treating Pose weight as 0.")

        # Filter dataset by license if requested
        valid_indices = np.arange(len(IDS))
        if require_public and not META.empty and "filename" in META.columns and "license" in META.columns:
            pd_mask = META["license"].astype(str).str.contains("Public Domain", case=False, na=False)
            pool_names = set(META.loc[pd_mask, "filename"].astype(str).apply(lambda s: Path(s).name))
            idx_mask = [i for i, fn in enumerate(IDS) if Path(fn).name in pool_names]
            if len(idx_mask) == 0:
                st.warning("No entries meet Public-Domain filter; ignoring filter.")
            else:
                valid_indices = np.array(idx_mask, dtype=int)

        # Prepare feature pools
        def take_rows(arr: Optional[np.ndarray], idxs: np.ndarray) -> Optional[np.ndarray]:
            if arr is None:
                return None
            return arr[idxs] if arr.shape[0] == len(IDS) else None

        # (Note: pools are not directly used below since we score against full arrays;
        # kept here if you later want to narrow scoring to 'valid_indices'.)

        # Rank across full set, then map indices if filtered:
        results = rank_results(
            q_clip=q_clip,
            q_pose=query_pose_feat,
            q_color=q_color,
            topk=min(top_k, len(valid_indices)),
            weights=(w_clip, w_pose_eff, w_color),
        )

        # Map to filtered indices if needed
        if len(valid_indices) != len(IDS):
            mapped = []
            for local_idx, sc in results:
                global_idx = int(valid_indices[local_idx])
                mapped.append((global_idx, sc))
            results = mapped

        dt = (time.time() - t0) * 1000
        st.info(f"Search time: {dt:.1f} ms")

        display_results(results, k=top_k)

# Tips
with st.expander("Tips"):
    st.markdown(
        """
- If you see **iCloud .icloud** placeholders, open the folder in Finder and choose **“Download Now”**.
- Adjust **CLIP / Pose / Color** weights in the sidebar to influence ranking.
- The **Public-Domain** filter works only if `license` exists in your metadata CSV.
- First run builds and caches dataset features in `.clip_cache/`.
- Pose detection uses `ultralytics` (YOLOv8n-pose). Ensure the model downloads once with internet access.
- If you only want CLIP + Color, set Pose weight to 0.
"""
    )
