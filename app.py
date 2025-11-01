# app.py
# Embodied Aesthetic Reconstruction — Streamlit App (Resonance-first UI)
# --------------------------------------------------------------------
# - CLIP + Pose + Color + Significance fusion
# - Human-centered UI: semantic "resonance" labels instead of raw numeric scores
# - Sidebar toggle "Debug mode" to show raw similarity when needed
# - Significance score (0–100) + badges + "Learn more"
# - Extra metadata fields: price_estimate_usd, significance_text, interpretive_note_cn
# - iCloud placeholder avoidance, robust pathing, cached embeddings
# - YOLOv8n-pose via HuggingFace (optional), proper device mapping for MPS/CUDA/CPU
# --------------------------------------------------------------------

from __future__ import annotations
import io
import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# -------------------- Torch / device --------------------
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

def get_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' on Nvidia, else 'cpu'."""
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
if TORCH_OK:
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

def yolo_device() -> str:
    """Map torch device to Ultralytics' expected device string."""
    if DEVICE == "mps":
        return "mps"
    if DEVICE == "cuda":
        return "0"
    return "cpu"

# -------------------- Paths --------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
INTERIM_DIR = DATA_DIR / "interim"
CACHE_DIR = APP_DIR / ".clip_cache"
CACHE_DIR.mkdir(exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = DATA_DIR / "portrait_works.csv"

# -------------------- UI config --------------------
st.set_page_config(
    page_title="Embodied Aesthetic Reconstruction",
    page_icon="🖼️",
    layout="wide",
)

# -------------------- Utils --------------------
def is_icloud_placeholder(p: Path) -> bool:
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Matching Weights")
    w_clip = st.slider("CLIP weight", 0.0, 1.0, 1.00, 0.05)
    w_pose = st.slider("Pose weight", 0.0, 1.0, 0.30, 0.05)
    w_color = st.slider("Color weight", 0.0, 1.0, 0.20, 0.05)
    w_sig  = st.slider("Significance weight", 0.0, 1.0, 0.20, 0.05)

    st.markdown("---")
    st.header("🔎 Filters")
    require_public = st.checkbox("Require Public-Domain license", value=False)

    st.markdown("---")
    st.header("🧪 Debug")
    show_debug = st.checkbox("Show raw similarity score", value=False)

    st.markdown("---")
    st.header("📊 Results")
    top_k = st.slider("Top-K artworks", 1, 20, 6, 1)

    st.markdown("---")
    st.header("💻 Active device")
    st.success(f"{DEVICE}")

    st.markdown("---")
    st.header("🧍 Overlay")
    overlay_skeleton = st.checkbox("Draw skeleton on preview (if available)", value=True)

# -------------------- Metadata & dataset --------------------
def load_metadata() -> pd.DataFrame:
    """
    Expected (optional) columns:
    filename,title,artist,year,license,museum,accession,movement,
    is_masterwork,citations,exhibitions,auction_price_usd,views_per_year,
    price_estimate_usd,significance_text,interpretive_note_cn,notable_tags,source_links
    """
    if METADATA_CSV.exists():
        try:
            return pd.read_csv(METADATA_CSV)
        except Exception as e:
            st.warning(f"Could not read metadata CSV ({METADATA_CSV.name}): {e}")
    # Fallback minimal schema
    return pd.DataFrame({
        "filename": [], "title": [], "artist": [], "year": [], "license": [],
        "museum": [], "accession": [], "movement": [], "is_masterwork": [],
        "citations": [], "exhibitions": [], "auction_price_usd": [], "views_per_year": [],
        "price_estimate_usd": [], "significance_text": [], "interpretive_note_cn": [],
        "notable_tags": [], "source_links": []
    })

META = load_metadata()

def list_dataset_images() -> List[Path]:
    if not IMAGES_DIR.exists():
        return []
    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files += list(IMAGES_DIR.glob(ext))
    files = [p for p in files if not is_icloud_placeholder(p)]
    return sorted(files)

DATASET_FILES = list_dataset_images()

# -------------------- Significance (0..100) --------------------
TOP_MUSEUMS = ["met", "national gallery", "tate", "louvre", "uffizi", "hermitage", "ng london"]

def _logn(x, d=1.0):
    try:
        return math.log10(max(float(x), 1.0)) / d
    except Exception:
        return 0.0

def compute_significance_row(row) -> float:
    """Heuristic significance score (0..100)."""
    w = dict(master=0.35, museum=0.20, cites=0.15, exhib=0.10, auction=0.10, views=0.10)
    score = 0.0
    if str(row.get("is_masterwork", "0")).lower() in ["1", "true", "yes", "y"]:
        score += w["master"]
    museum = str(row.get("museum", "")).lower()
    if museum:
        score += w["museum"] * (1.0 if any(m in museum for m in TOP_MUSEUMS) else 0.5)
    score += w["cites"]   * _logn(row.get("citations", 0),        d=3.0)
    score += w["exhib"]   * _logn(row.get("exhibitions", 0),      d=2.0)
    score += w["auction"] * _logn(row.get("auction_price_usd",0), d=8.0)
    score += w["views"]   * _logn(row.get("views_per_year", 0),   d=6.0)
    return float(np.clip(score, 0.0, 1.0) * 100.0)

def significance_badges(row) -> List[str]:
    badges = []
    if str(row.get("is_masterwork","0")).lower() in ["1","true","yes","y"]:
        badges.append("Masterwork")
    if str(row.get("museum","")).strip():
        badges.append("Permanent Collection")
    try:
        if int(row.get("citations", 0)) >= 50:
            badges.append("Canon")
    except Exception:
        pass
    return badges[:3]

def filename_key(path_or_name: str) -> str:
    return Path(path_or_name).name

META_BY_NAME = {}
if not META.empty and "filename" in META.columns:
    for _, r in META.iterrows():
        META_BY_NAME[filename_key(str(r.get("filename","")))] = r

# -------------------- OpenCLIP --------------------
@st.cache_resource(show_spinner=False)
def load_openclip():
    try:
        import open_clip
    except Exception as e:
        st.error(f"open_clip_torch not installed: {e}")
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

# -------------------- YOLO weights via HuggingFace (optional) --------------------
def get_yolo_pose_weights_path() -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return "yolov8n-pose.pt"
    try:
        local = hf_hub_download(
            repo_id="ultralytics/yolov8n-pose",
            filename="yolov8n-pose.pt",
            local_dir=str(Path.home() / ".cache" / "hf"),
            local_dir_use_symlinks=False
        )
        return local
    except Exception:
        return "yolov8n-pose.pt"

# -------------------- Feature extraction & cache --------------------
EMB_PATH   = CACHE_DIR / "embeddings.npy"
IDS_PATH   = CACHE_DIR / "ids.json"
POSE_PATH  = CACHE_DIR / "pose.npy"
COLOR_PATH = CACHE_DIR / "color.npy"
SIG_PATH   = CACHE_DIR / "significance.npy"

def _color_feature(im: Image.Image) -> np.ndarray:
    im = im.resize((256, 256), Image.BILINEAR)
    arr = np.array(im.convert("HSV"))
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    hist_h, _ = np.histogram(h, bins=32, range=(0, 255), density=True)
    hist_s, _ = np.histogram(s, bins=16, range=(0, 255), density=True)
    hist_v, _ = np.histogram(v, bins=8,  range=(0, 255), density=True)
    feat = np.concatenate([hist_h, hist_s, hist_v]).astype("float32")
    return feat / (np.linalg.norm(feat) + 1e-8)

def _pose_feature_from_kpts(kpts_xyv: np.ndarray) -> np.ndarray:
    if kpts_xyv is None or len(kpts_xyv) == 0:
        return np.zeros(34, dtype="float32")
    xy = kpts_xyv[:, :2]
    min_xy = xy.min(0); max_xy = xy.max(0)
    wh = np.maximum(max_xy - min_xy, 1e-6)
    xy_n = (xy - min_xy) / wh
    feat = xy_n.reshape(-1).astype("float32")
    if feat.shape[0] < 34:
        feat = np.pad(feat, (0, 34 - feat.shape[0]))
    else:
        feat = feat[:34]
    return feat / (np.linalg.norm(feat) + 1e-8)

def _detect_pose_pil(im: Image.Image) -> Optional[np.ndarray]:
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        weights = get_yolo_pose_weights_path()
        model = YOLO(weights)
        arr = np.array(im.convert("RGB"))
        res = model.predict(
            source=arr, imgsz=512, conf=0.25, verbose=False, device=yolo_device()
        )
        if not res or len(res[0].keypoints) == 0:
            return None
        kpts = res[0].keypoints.xy.cpu().numpy()
        if kpts.ndim == 4: kpts = kpts[0]
        if kpts.ndim == 3: k = kpts[0]
        else: return None
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
def build_dataset_embeddings(files: List[Path]):
    """Return (clip_embeds, ids, pose_feats, color_feats, signif_norm)."""
    ids = [str(p.relative_to(APP_DIR)) for p in files]

    # Use cache if aligned
    if EMB_PATH.exists() and IDS_PATH.exists() and SIG_PATH.exists():
        try:
            cached_ids = json.loads(IDS_PATH.read_text())
            if cached_ids == ids:
                clip_arr  = np.load(EMB_PATH)
                pose_arr  = np.load(POSE_PATH)  if POSE_PATH.exists()  else None
                color_arr = np.load(COLOR_PATH) if COLOR_PATH.exists() else None
                sig_arr   = np.load(SIG_PATH)
                return clip_arr, ids, pose_arr, color_arr, sig_arr
        except Exception:
            pass

    clip_feats, pose_feats, color_feats, signif_vals = [], [], [], []

    if MODEL_CLIP is None:
        st.error("OpenCLIP not ready; cannot build embeddings.")
        return np.zeros((0, 512), dtype="float32"), ids, None, None, np.zeros((0,), dtype="float32")

    progress = st.progress(0.0, text="Building dataset embeddings/features…")
    for i, p in enumerate(files):
        progress.progress((i + 1) / max(1, len(files)))
        im = load_image_safe(p)

        # CLIP
        if im is not None:
            try:
                im_clip = center_crop_long_edge(im, 224)
                x = tensor_from_pil_clip(im_clip)
                with torch.no_grad():
                    z = MODEL_CLIP.encode_image(x)
                    z = z / z.norm(dim=-1, keepdim=True)
                clip_feats.append(z.cpu().numpy().astype("float32")[0])
            except Exception:
                clip_feats.append(np.zeros(512, dtype="float32"))
        else:
            clip_feats.append(np.zeros(512, dtype="float32"))

        # Color
        if im is not None:
            try:
                color_feats.append(_color_feature(im))
            except Exception:
                color_feats.append(np.zeros(56, dtype="float32"))
        else:
            color_feats.append(np.zeros(56, dtype="float32"))

        # Pose
        pf = _dataset_pose_feature(p)
        pose_feats.append(pf if pf is not None else np.zeros(34, dtype="float32"))

        # Significance prior from metadata (0..1)
        fname = filename_key(p.name)
        row = META_BY_NAME.get(fname)
        sig = compute_significance_row(row) / 100.0 if row is not None else 0.0
        signif_vals.append(float(sig))

    clip_arr  = np.vstack(clip_feats).astype("float32") if clip_feats else np.zeros((0, 512), dtype="float32")
    pose_arr  = np.vstack(pose_feats).astype("float32") if pose_feats else None
    color_arr = np.vstack(color_feats).astype("float32") if color_feats else None
    sig_arr   = np.array(signif_vals, dtype="float32")

    # Cache
    try:
        np.save(EMB_PATH, clip_arr)
        np.save(POSE_PATH, pose_arr)
        np.save(COLOR_PATH, color_arr)
        np.save(SIG_PATH,  sig_arr)
        IDS_PATH.write_text(json.dumps(ids))
    except Exception:
        pass

    return clip_arr, ids, pose_arr, color_arr, sig_arr

# ---- Build dataset features ----
if len(DATASET_FILES) == 0:
    st.error(f"❌ No images found or folder is empty: {IMAGES_DIR}\nMake sure iCloud files are downloaded locally (no .icloud).")
else:
    st.success(f"✅ Dataset images: {len(DATASET_FILES)}")

CLIP_DS, IDS, POSE_DS, COLOR_DS, SIG_DS = build_dataset_embeddings(DATASET_FILES)  # SIG_DS in [0,1]

# ---- Ready checks ----
READY = True
reasons = []
if MODEL_CLIP is None:
    READY = False; reasons.append("OpenCLIP not loaded.")
if len(DATASET_FILES) == 0:
    READY = False; reasons.append("No images in data/images (or still .icloud).")
if CLIP_DS is None or getattr(CLIP_DS, "shape", (0,))[0] == 0:
    READY = False; reasons.append("Dataset embeddings not built yet.")
if not READY:
    st.warning("Matching is not ready: " + " ".join(reasons))
else:
    st.success("Matching is ready ✔")

# -------------------- Similarity + Resonance --------------------
def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a is None or b is None:
        return None
    if a.ndim == 1:
        a = a[None, :]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a @ b.T)

def interpret_score(score: float) -> str:
    """Map numeric similarity to a poetic/semantic label."""
    if score > 0.80: return "Strong resonance 💫"
    if score > 0.65: return "Aesthetic kinship ✨"
    if score > 0.50: return "Subtle correspondence 🌙"
    return "Distant echo 🌫️"

# -------------------- Pose overlay (preview only) --------------------
def draw_skeleton_overlay(im: Image.Image) -> Tuple[Image.Image, Optional[np.ndarray]]:
    try:
        from ultralytics import YOLO
        import cv2
    except Exception:
        return im, None

    weights = get_yolo_pose_weights_path()
    model = YOLO(weights)
    arr = np.array(im.convert("RGB"))
    res = model.predict(source=arr, imgsz=512, conf=0.25, verbose=False, device=yolo_device())
    if not res or len(res[0].keypoints) == 0:
        return im, None

    canvas = arr.copy()
    pts = res[0].keypoints.xy.cpu().numpy()
    if pts.ndim == 4: pts = pts[0]
    if pts.ndim == 3: pts = pts[0]

    PAIRS = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12),(5,11),(6,12)]
    try:
        import cv2
        for (a,b) in PAIRS:
            xa,ya = pts[a]; xb,yb = pts[b]
            cv2.line(canvas,(int(xa),int(ya)),(int(xb),int(yb)),(0,255,0),3)
        for (x,y) in pts:
            cv2.circle(canvas,(int(x),int(y)),4,(0,0,255),-1)
        im_draw = Image.fromarray(canvas)
    except Exception:
        im_draw = im

    v = np.ones((pts.shape[0],1),dtype="float32")
    kxyv = np.concatenate([pts.astype("float32"), v], axis=1)
    pf = _pose_feature_from_kpts(kxyv)
    return im_draw, pf

# -------------------- UI — Main --------------------
st.title("Embodied Aesthetic Reconstruction")
st.caption("Camera / Upload → CLIP + Pose + Color + Significance → Resonant artworks")

status_cols = st.columns(3)
with status_cols[0]:
    st.success(f"Images dir: {IMAGES_DIR.relative_to(APP_DIR)}" if IMAGES_DIR.exists() else f"Missing: {IMAGES_DIR}")
with status_cols[1]:
    st.info(f"Dataset files: {len(DATASET_FILES)}")
with status_cols[2]:
    st.info(f"Models: OpenCLIP={'✔︎' if MODEL_CLIP else '✖︎'} · Pose(ultralytics)={'✔︎'}")

st.markdown("---")

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

st.markdown("---")
run = st.button("🔎 Run Matching", disabled=not READY)

# -------------------- Display helpers --------------------
def display_results(idx_scores: List[Tuple[int, float]], k: int):
    if len(idx_scores) == 0:
        st.warning("No results to display.")
        return
    cols = st.columns(min(3, k))
    for i, (idx, sc) in enumerate(idx_scores[:k]):
        fn_rel = IDS[idx]
        p = (APP_DIR / fn_rel).resolve()
        im = load_image_safe(p)

        row = META_BY_NAME.get(filename_key(p.name))
        if row is not None:
            title  = str(row.get("title", "")).strip()
            artist = str(row.get("artist", "")).strip()
            year   = str(row.get("year", "")).strip()
            museum = str(row.get("museum", "")).strip()
            price  = str(row.get("price_estimate_usd", "")).strip()
            sigtxt = str(row.get("significance_text", "")).strip()
            sigcn  = str(row.get("interpretive_note_cn", "")).strip()
            links  = str(row.get("source_links", "")).strip()
            sig100 = compute_significance_row(row)
            badges = significance_badges(row)
        else:
            title=artist=year=museum=price=sigtxt=sigcn=links=""
            sig100, badges = 0.0, []

        with cols[i % len(cols)]:
            if im is not None:
                st.image(im, use_container_width=True)

            # Resonance-first display
            if show_debug:
                st.markdown(f"**Score:** {sc:.3f}")
            else:
                st.markdown(f"**{interpret_score(sc)}**")

            if title or artist or year:
                st.caption(f"{title} — {artist} ({year})")
            if museum:
                st.caption(f"🏛️ {museum}")
            if price:
                st.caption(f"💰 Estimated value: ${price}")

            st.markdown(f"**Significance:** {sig100:.0f}/100")
            if badges:
                st.caption(" · ".join(badges))
            if sigtxt:
                st.markdown(f"_{sigtxt}_")
            if sigcn:
                st.markdown(f"**Curatorial note (CN):** {sigcn}")
            if links:
                first = links.split(";")[0].strip()
                if first:
                    st.write(f"[Learn more]({first})")

# -------------------- Matching --------------------
if run:
    if query_img is None:
        st.warning("Please take a photo or upload an image first.")
    else:
        st.write("Computing embeddings…")
        t0 = time.time()

        q_clip = None
        if MODEL_CLIP is not None:
            try:
                buf = io.BytesIO()
                query_img.save(buf, format="JPEG", quality=92)
                q_clip = embed_image_clip(buf.getvalue())
            except Exception:
                q_clip = None

        try:
            q_color = _color_feature(query_img)
        except Exception:
            q_color = None

        if query_pose_feat is None and w_pose > 0:
            query_pose_feat = _detect_pose_pil(center_crop_long_edge(query_img, 512))
        w_pose_eff = w_pose if query_pose_feat is not None and POSE_DS is not None else 0.0
        if w_pose > 0 and w_pose_eff == 0.0:
            st.info("No pose feature detected or dataset pose not precomputed; treating Pose weight as 0.")

        valid_indices = np.arange(len(IDS))
        if require_public and not META.empty and {"filename", "license"}.issubset(META.columns):
            pd_mask = META["license"].astype(str).str.contains("Public Domain", case=False, na=False)
            pool_names = set(META.loc[pd_mask, "filename"].astype(str).apply(lambda s: Path(s).name))
            filt = [i for i, fn in enumerate(IDS) if Path(fn).name in pool_names]
            if len(filt) > 0:
                valid_indices = np.array(filt, dtype=int)
            else:
                st.warning("No entries meet Public-Domain filter; ignoring filter.")

        def slicer(arr):
            if arr is None or arr.shape[0] != len(IDS):
                return None
            return arr[valid_indices]

        CLIP_POOL  = slicer(CLIP_DS)
        POSE_POOL  = slicer(POSE_DS)
        COLOR_POOL = slicer(COLOR_DS)
        SIG_POOL   = SIG_DS[valid_indices] if SIG_DS is not None and SIG_DS.shape[0] == len(IDS) else None

        def cos_sim_local(q, bank):
            if q is None or bank is None:
                return None
            a = q[None, :] if q.ndim == 1 else q
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            b = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
            return (a @ b.T)[0]

        score = np.zeros(valid_indices.size, dtype="float32")
        s = cos_sim_local(q_clip, CLIP_POOL)
        if s is not None: score += w_clip * s
        s = cos_sim_local(query_pose_feat, POSE_POOL) if w_pose_eff > 0 else None
        if s is not None: score += w_pose_eff * s
        s = cos_sim_local(q_color, COLOR_POOL)
        if s is not None: score += w_color * s
        if SIG_POOL is not None and w_sig > 0.0:
            score += w_sig * SIG_POOL

        topk_local = np.argsort(-score)[:min(top_k, score.size)]
        results = [(int(valid_indices[i]), float(score[i])) for i in topk_local]

        dt = (time.time() - t0) * 1000
        st.info(f"Search time: {dt:.1f} ms")

        display_results(results, k=top_k)

# -------------------- Tips --------------------
with st.expander("Tips"):
    st.markdown(
        """
- If you see **iCloud .icloud** placeholders, open the folder in Finder and choose **“Download Now”**.
- Adjust **CLIP / Pose / Color / Significance** weights to influence ranking.
- **Debug mode** reveals raw similarity scores; keep it off for exhibitions to maintain a poetic tone.
- The **Public-Domain** filter works only if `license` exists in your metadata CSV.
- First run builds and caches dataset features in `.clip_cache/`.
- Pose detection uses `ultralytics` (YOLOv8n-pose). Weights are pulled from Hugging Face when possible.
- *Significance* is a heuristic combining museum presence, citations, and attention. It reflects historical canons yet encourages plural aesthetics.
"""
    )
