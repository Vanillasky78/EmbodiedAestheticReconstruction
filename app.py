# app.py
import os
import tempfile
from PIL import Image
import streamlit as st

# ---- project modules ----
from processors.retrieval import load_clip, embed_image, CorpusIndex
from processors.pose import extract_pose
from processors.visual import (
    procrustes_pose_score,
    face_angle_score,
    silhouette_ssim,
    color_corr,
)
from processors.matching import (
    composition_score,
    palette_score,
    limb_hist_score,
)
from processors.fusion_advanced import integrate_into_artwork


# ---------------------- index loading ----------------------
@st.cache_resource
def load_indices():
    """Discover and load available indices (museum + fashion brands)."""
    ROOT = os.path.dirname(os.path.abspath(__file__))
    inds = {}

    # Museum
    p_museum = os.path.join(ROOT, "data", "indices", "paintings")
    if os.path.exists(p_museum):
        try:
            inds[("Museum", "All")] = CorpusIndex(p_museum)
        except Exception:
            pass

    # Fashion: data/indices_fashion/<Brand>/
    fashion_root = os.path.join(ROOT, "data", "indices_fashion")
    if os.path.isdir(fashion_root):
        for brand in sorted(os.listdir(fashion_root)):
            p = os.path.join(fashion_root, brand)
            if os.path.isdir(p):
                try:
                    inds[("Fashion", brand)] = CorpusIndex(p)
                except Exception:
                    pass
    return inds


# ---------------------- model warmup ----------------------
st.set_page_config(page_title="Embodied Aesthetic — Accurate Match & Fusion", layout="wide")
st.title("Embodied Aesthetic — Accurate Matching & Scene-Aware Fusion")

with st.spinner("Loading CLIP and indices..."):
    model, preprocess, device = load_clip()
    INDICES = load_indices()


# ---------------------- UI layout ----------------------
colL, colM, colR = st.columns([1.1, 1.5, 1.4])

# ========== Left: input & weights ==========
with colL:
    st.subheader("1) Input & Controls")

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

    domain = st.selectbox("Domain", ["Museum", "Fashion"])
    brands_available = ["All"] if domain == "Museum" else sorted(
        [b for (d, b) in INDICES.keys() if d == "Fashion"]
    )
    # In case no fashion indices exist yet
    if domain == "Fashion" and not brands_available:
        st.info("No fashion indices found in data/indices_fashion. Build one first.")
        brands_available = ["<None>"]
    brand = st.selectbox("Brand", brands_available, index=0)

    topk = st.slider("Top-K", 1, 12, 6)

    st.markdown("**Re-ranking Weights**")
    # Base signals
    w_clip  = st.slider("w_clip (semantic)",    0.0, 1.0, 0.40, 0.05)
    w_pose  = st.slider("w_pose (pose)",        0.0, 1.0, 0.20, 0.05)
    w_face  = st.slider("w_face (orientation)", 0.0, 1.0, 0.15, 0.05)
    w_sil   = st.slider("w_sil (silhouette)",   0.0, 1.0, 0.10, 0.05)
    w_color = st.slider("w_color (color)",      0.0, 1.0, 0.05, 0.05)
    # New signals
    w_comp  = st.slider("w_comp (composition)", 0.0, 1.0, 0.05, 0.05)
    w_palet = st.slider("w_palette (palette)",  0.0, 1.0, 0.03, 0.05)
    w_limb  = st.slider("w_limb (limb angles)", 0.0, 1.0, 0.02, 0.05)

    find_btn = st.button("Find Matches", type="primary")


# ========== Middle: retrieval & multi-signal re-ranking ==========
results = []
qimg = None
with colM:
    st.subheader("2) Matches (Multi-signal Re-ranked)")

    key = (domain, brand if domain == "Fashion" else "All")
    index_ready = key in INDICES

    if not index_ready:
        st.warning("Index not available for this domain/brand. Please build it first.")
    elif uploaded and find_btn:
        # 1) read query
        qimg = Image.open(uploaded).convert("RGB")
        st.image(qimg, caption="Query", use_container_width=True)

        # 2) search by CLIP
        idx = INDICES[key]
        qfeat = embed_image(model, preprocess, device, qimg)
        sims, ids = idx.search(qfeat, topk=max(topk * 5, topk))  # expand candidate pool

        # 3) pose for query (for pose/limb features)
        qpose = extract_pose(qimg)

        # 4) compute multi-signal scores for candidates
        re_rank = []
        for s, i in zip(sims, ids):
            path = idx.paths[i]
            clip_s = float(s)

            # pose similarity (Procrustes)
            pose_s = None
            if qpose.ok and idx.poses[i] is not None:
                pose_s = procrustes_pose_score(qpose.keypoints, idx.poses[i])

            # read candidate as PIL once for visual signals
            ref_pil = Image.open(path).convert("RGB")

            # other cues
            face_s  = face_angle_score(qimg, ref_pil)           # 0..1
            sil_s   = silhouette_ssim(qimg, ref_pil)            # 0..1
            color_s = color_corr(qimg, ref_pil)                 # 0..1
            comp_s  = composition_score(qimg, ref_pil)          # 0..1
            pal_s   = palette_score(qimg, ref_pil, k=5)         # 0..1
            limb_s  = limb_hist_score(qpose.keypoints, idx.poses[i]) if (qpose.ok and idx.poses[i] is not None) else 0.0

            # fused score
            final = (
                w_clip  * clip_s
              + w_pose  * (pose_s or 0.0)
              + w_face  * face_s
              + w_sil   * sil_s
              + w_color * color_s
              + w_comp  * comp_s
              + w_palet * pal_s
              + w_limb  * limb_s
            )

            re_rank.append((path, clip_s, pose_s, face_s, sil_s, color_s, comp_s, pal_s, limb_s, final))

        # 5) sort & display
        re_rank.sort(key=lambda x: x[-1], reverse=True)
        results = re_rank[:topk]

        for (path, cs, ps, fs, ss, cc, cps, pls, lbs, fin) in results:
            cap = f"Fused={fin:.3f} | CLIP={cs:.3f}"
            if ps is not None:
                cap += f" | Pose={ps:.3f}"
            cap += f" | Face={fs:.3f} | Sil={ss:.3f} | Color={cc:.3f} | Comp={cps:.3f} | Pal={pls:.3f} | Limb={lbs:.3f}"
            st.image(path, caption=cap, use_container_width=True)


# ========== Right: Scene-aware fusion ==========
with colR:
    st.subheader("3) Fusion Result (Scene-aware)")
    if qimg and results:
        # choose top-1 match for fusion
        ref_path = results[0][0]

        # Write uploaded image to a real temp file so OpenCV-based code can read it
        tmp_q = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.open(uploaded).convert("RGB").save(tmp_q.name)

        # Scene-aware integration (advanced)
        fused = integrate_into_artwork(tmp_q.name, ref_path)
        st.image(fused, caption="Integrated into Artwork", use_container_width=True)

        # Optional tip
        st.caption("Tip: tune weights on the left (pose/face/silhouette/composition/palette) if the match still feels off.")
