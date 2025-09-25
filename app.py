from pathlib import Path
import io
import numpy as np
import streamlit as st
from PIL import Image

from core.pose import pose_to_embedding
from core.personalise import personality_vector, fuse_embeddings
from core.style_select import pick_style
from core.image_ops import stylise_image
from audio.synth import embedding_to_tones
from ui.components import sidebar_personality, sidebar_controls, save_image, save_audio
from ethics.consent import consent_text

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Embodied Aesthetic Reconstruction — MVP", layout="wide")
st.title("Embodied Aesthetic Reconstruction — MVP")
st.caption("Modular prototype: pose → embedding → personalised style → image + audio.")
st.markdown(consent_text())

colL, colR = st.columns([1, 1])

with colL:
    st.header("1) Input")
    src = st.radio("Choose input", ["Upload Image", "Use Demo"], horizontal=True)
    if src == "Upload Image":
        up = st.file_uploader("Upload a photo (full-body preferred)", ["png", "jpg", "jpeg"])
        img = Image.open(up) if up else None
    else:
        # 简易占位图
        import numpy as np, cv2
        demo = np.full((512, 384, 3), 240, np.uint8)
        cv2.putText(demo, "Demo Figure", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        img = Image.fromarray(demo)

    if img is not None:
        st.image(img, caption="Input image", use_column_width=True)
    else:
        st.info("Please upload an image.")

with colR:
    st.header("2) Controls")
    traits = sidebar_personality()
    style_strength, consent, save_outputs = sidebar_controls()
    ready = st.button("Generate", type="primary")

st.markdown("---")

if ready:
    if not consent:
        st.warning("Please provide consent in the sidebar to proceed.")
        st.stop()
    if img is None:
        st.error("No image provided.")
        st.stop()

    with st.spinner("Extracting pose and computing embedding…"):
        pose_emb = pose_to_embedding(img)
        pers_vec = personality_vector(traits)               # ← 修正：不要用 collect_personality_vector
        fused = fuse_embeddings(pose_emb, pers_vec)
        style_name, sim = pick_style(fused)

    st.subheader("3) Style Selection")
    st.write(f"**Chosen style:** {style_name}  |  Cosine match: **{sim:.3f}**")

    with st.spinner("Stylising image…"):
        out_img = stylise_image(img, style_name, style_strength)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(out_img, caption=f"Stylised: {style_name}", use_column_width=True)
    with c2:
        energy = float(np.mean(np.abs(fused))) if fused.size else 0.0
        st.metric("Embedding energy", f"{energy:.3f}")
        st.caption("Higher energy → richer harmonics in audio.")

    with st.spinner("Synthesising audio…"):
        wav_bytes = embedding_to_tones(fused, dur_s=3.0)
    st.audio(wav_bytes, format="audio/wav")

    # Save
    if save_outputs:
        p1 = save_image(out_img, OUTPUT_DIR, prefix="render")
        p2 = save_audio(wav_bytes, OUTPUT_DIR, prefix="audio")
        st.success(f"Saved image → {p1}\nSaved audio → {p2}")

    # Download buttons（修正：正确编码 PNG）
    png_buf = io.BytesIO()
    out_img.save(png_buf, format="PNG")
    png_buf.seek(0)
    st.download_button("Download stylised image",
                       data=png_buf.getvalue(),
                       file_name="stylised.png",
                       mime="image/png")
    st.download_button("Download audio (WAV)",
                       data=wav_bytes,
                       file_name="audio.wav",
                       mime="audio/wav")

st.markdown("---")
with st.expander("Next steps (how to swap in your real models)"):
    st.markdown(
        """
**Replaceable modules**
- `stylise_image(...)` → swap in Diffusion / LoRA / DreamBooth inference.
- `embedding_to_tones(...)` → route to RAVE / AudioLM (pose-controlled).
- `fuse_embeddings(...)` → learn a better projection from personality/affect.
- `pose_to_embedding(...)` → extend to video (temporal energy).

**Ethics hooks**
- Keep the consent gate; add a data retention toggle.
- Provide a “de-bias” switch and dataset/model cards.
"""
    )
