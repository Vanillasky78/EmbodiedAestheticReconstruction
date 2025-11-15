# frontend/app_frontend.py
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configuration
API_URL = "http://127.0.0.1:8000/match"
YOLO_MODEL_PATH = "models/yolov8n-pose.pt"

# Load YOLO pose model
pose_model = YOLO(YOLO_MODEL_PATH)

st.set_page_config(layout="wide")
st.title("🎨 Real-Time Pose-to-Art Matching")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Camera View with Pose Skeleton")
    camera = st.camera_input("Take a photo")

    if camera is not None:
        file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO pose detection
        results = pose_model(frame, verbose=False)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Pose Detection")

        # Send to API
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = st.experimental_memo(lambda: img_pil, ttl=60)(img_pil)

        with st.spinner("Matching artwork..."):
            _, img_encoded = cv2.imencode(".jpg", frame)
            files = {"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            response = requests.post(API_URL, files=files, data={"museum": "local"})

        if response.status_code == 200:
            matches = response.json().get("results", [])
        else:
            matches = []
            st.warning("Failed to retrieve matches.")

with col2:
    st.header("🖼️ Matched Artworks")
    if camera is None:
        st.info("Waiting for photo input...")
    else:
        for match in matches:
            filename = match.get("filename")
            score = match.get("score")
            artist = match.get("artist", "Unknown")
            year = match.get("year", "Unknown")
            price = match.get("auction_price_usd") or match.get("price_estimate_usd") or "N/A"

            # Layout formatting
            st.image(f"data/local/images/{filename}", width=300)
            st.markdown(f"<div style='background-color:#ffdd00; font-family:Courier; font-size:22px; padding:6px'>💰 {price}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#ffdd00; font-family:Courier; font-size:18px; padding:4px'>📅 {year}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#ffdd00; font-family:Courier; font-size:20px; padding:5px'>🎨 {artist}</div>", unsafe_allow_html=True)
            st.markdown(f"<hr style='margin-top:20px;margin-bottom:20px;'>", unsafe_allow_html=True)