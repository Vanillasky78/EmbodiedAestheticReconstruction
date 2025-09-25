# ui/components.py
import time
from pathlib import Path
import streamlit as st
from PIL import Image

def sidebar_personality():
    st.sidebar.subheader("Personalisation")
    defaults = [
        ("Openness", 0.7),
        ("Conscientiousness", 0.5),
        ("Extraversion", 0.5),
        ("Agreeableness", 0.5),
        ("Neuroticism", 0.3),
    ]
    vals = [(n, st.sidebar.slider(n, 0.0, 1.0, float(v), 0.01)) for n, v in defaults]
    return vals

def sidebar_controls():
    st.sidebar.subheader("Generation Controls")
    style_strength = st.sidebar.slider("Style Strength", 0.1, 1.0, 0.75, 0.01)
    consent = st.sidebar.checkbox(
        "I consent to process my image for research/artistic purposes.", value=False
    )
    save_outputs = st.sidebar.checkbox("Save outputs to ./outputs", value=True)
    return style_strength, consent, save_outputs

def save_image(img: Image.Image, outdir: Path, prefix: str) -> str:
    ts = int(time.time())
    path = outdir / f"{prefix}_{ts}.png"
    img.save(path)
    return str(path)

def save_audio(wav_bytes: bytes, outdir: Path, prefix: str) -> str:
    ts = int(time.time())
    path = outdir / f"{prefix}_{ts}.wav"
    path.write_bytes(wav_bytes)
    return str(path)
