# frontend/app_frontend.py
import streamlit as st
import requests
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("Embodied Aesthetic Retrieval — Final Presentation")

# Load YOLO model
model = YOLO("yolov8n-pose.pt")

# Sidebar options
with st.sidebar:
    st.header("Camera Input")
    use_camera = st.checkbox("Enable Camera", value=True)
    topk = st.slider("Top-k Matches", 1, 5, 3)
    museum = st.selectbox("Museum Scope", ["mixed", "local", "met"])

col1, col2 = st.columns(2)

# Camera section
with col1:
    st.markdown("### Live Camera Pose")
    frame_placeholder = st.empty()
    image_upload = None

    if use_camera:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            results = model(frame, verbose=False)[0]
            for pose in results.keypoints.xy:
                for x, y in pose:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_upload = Image.fromarray(rgb)
            frame_placeholder.image(rgb, channels="RGB", use_column_width=True)
        else:
            st.warning("Unable to read from camera.")

    uploaded = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image_upload = Image.open(uploaded).convert("RGB")
        frame_placeholder.image(image_upload, use_column_width=True)

# Matching section
with col2:
    st.markdown("### Matching Results")

    if image_upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_upload.save(tmp.name)
            with open(tmp.name, "rb") as f:
                files = {"image": f}
                data = {"museum": museum, "topk": str(topk)}
                res = requests.post("http://127.0.0.1:8000/match", data=data, files=files)

        if res.status_code == 200:
            results = res.json().get("results", [])
            for result in results:
                filename = result.get("filename", "")
                score = result.get("score", 0)
                artist = result.get("artist") or ""
                title = result.get("title") or ""
                year = result.get("year") or ""
                price = result.get("price_text") or ""

                st.image(f"data/local/images/{filename}", use_column_width=True)
                st.markdown(
                    f"<div style='background-color:yellow; padding:6px'>"
                    f"<span style='font-family:Courier; font-size:22px'><b>{price}</b></span><br>"
                    f"<span style='font-family:Courier; font-size:16px'>{year}</span><br>"
                    f"<span style='font-family:Courier; font-size:20px'><b>{artist}</b></span><br>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            st.error("Match request failed.")